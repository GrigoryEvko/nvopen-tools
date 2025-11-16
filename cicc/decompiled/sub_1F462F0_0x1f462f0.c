// Function: sub_1F462F0
// Address: 0x1f462f0
//
__int64 __fastcall sub_1F462F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdi
  unsigned int v5; // edx
  __int64 *v6; // rcx
  __int64 v7; // r8
  int v9; // ecx
  int v10; // r10d

  v2 = *(_QWORD *)(a1 + 216);
  v3 = *(unsigned int *)(v2 + 24);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(v2 + 8);
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 24LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 24 * v3) )
        return v6[1];
    }
    else
    {
      v9 = 1;
      while ( v7 != -4 )
      {
        v10 = v9 + 1;
        v5 = (v3 - 1) & (v9 + v5);
        v6 = (__int64 *)(v4 + 24LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        v9 = v10;
      }
    }
  }
  return a2;
}
