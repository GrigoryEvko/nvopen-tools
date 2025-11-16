// Function: sub_2AB16F0
// Address: 0x2ab16f0
//
__int64 __fastcall sub_2AB16F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // rdi
  int v8; // edx
  int v9; // r10d
  __int64 v10; // [rsp+0h] [rbp-8h]

  v2 = *(unsigned int *)(a1 + 232);
  v3 = *(_QWORD *)(a1 + 216);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
    {
LABEL_3:
      if ( v5 != (__int64 *)(v3 + 16 * v2) )
      {
        BYTE4(v10) = 1;
        LODWORD(v10) = *((_DWORD *)v5 + 2);
        return v10;
      }
    }
    else
    {
      v8 = 1;
      while ( v6 != -4096 )
      {
        v9 = v8 + 1;
        v4 = (v2 - 1) & (v8 + v4);
        v5 = (__int64 *)(v3 + 16LL * v4);
        v6 = *v5;
        if ( a2 == *v5 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
  BYTE4(v10) = 0;
  return v10;
}
