// Function: sub_1E80C40
// Address: 0x1e80c40
//
__int64 __fastcall sub_1E80C40(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r8
  int v4; // eax
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // r9
  int v9; // edx
  int v10; // r10d

  v2 = *(_DWORD *)(*a1 + 400LL);
  if ( v2 )
  {
    v3 = *(_QWORD *)(*a1 + 384LL);
    v4 = v2 - 1;
    v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      v2 = *((_DWORD *)v6 + 3) + *((_DWORD *)v6 + 2);
    }
    else
    {
      v9 = 1;
      while ( v7 != -8 )
      {
        v10 = v9 + 1;
        v5 = v4 & (v9 + v5);
        v6 = (__int64 *)(v3 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        v9 = v10;
      }
      v2 = 0;
    }
  }
  return (unsigned int)(*(_DWORD *)(a1[1] + 36LL) - v2);
}
