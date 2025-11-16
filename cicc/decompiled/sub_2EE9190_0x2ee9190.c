// Function: sub_2EE9190
// Address: 0x2ee9190
//
__int64 __fastcall sub_2EE9190(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r8
  int v3; // eax
  int v4; // eax
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // r9
  int v9; // edx
  int v10; // r10d

  v2 = *(_QWORD *)(*a1 + 384LL);
  v3 = *(_DWORD *)(*a1 + 400LL);
  if ( v3 )
  {
    v4 = v3 - 1;
    v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v2 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      v3 = *((_DWORD *)v6 + 2) + *((_DWORD *)v6 + 3);
    }
    else
    {
      v9 = 1;
      while ( v7 != -4096 )
      {
        v10 = v9 + 1;
        v5 = v4 & (v9 + v5);
        v6 = (__int64 *)(v2 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        v9 = v10;
      }
      v3 = 0;
    }
  }
  return (unsigned int)(*(_DWORD *)(a1[1] + 36LL) - v3);
}
