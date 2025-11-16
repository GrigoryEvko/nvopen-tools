// Function: sub_1458650
// Address: 0x1458650
//
__int64 __fastcall sub_1458650(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  unsigned int v4; // edx
  __int64 *v5; // rcx
  __int64 v6; // r8
  int v8; // ecx
  int v9; // r10d

  v2 = *(unsigned int *)(a1 + 208);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 192);
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 56LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
    {
LABEL_3:
      if ( v5 != (__int64 *)(v3 + 56 * v2) )
        return v5[1];
    }
    else
    {
      v8 = 1;
      while ( v6 != -8 )
      {
        v9 = v8 + 1;
        v4 = (v2 - 1) & (v8 + v4);
        v5 = (__int64 *)(v3 + 56LL * v4);
        v6 = *v5;
        if ( a2 == *v5 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
  return 0;
}
