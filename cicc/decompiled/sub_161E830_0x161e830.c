// Function: sub_161E830
// Address: 0x161e830
//
__int64 __fastcall sub_161E830(__int64 a1)
{
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rsi
  int v5; // r8d
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  int v11; // edx
  int v12; // r10d

  v2 = ***(_QWORD ***)a1;
  v3 = *(_DWORD *)(v2 + 456);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a1 + 24);
    v5 = v3 - 1;
    v6 = *(_QWORD *)(v2 + 440);
    v7 = (v3 - 1) & (((unsigned int)*(_QWORD *)(a1 + 24) >> 9) ^ ((unsigned int)v4 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
    {
LABEL_3:
      *v8 = -8;
      --*(_DWORD *)(v2 + 448);
      ++*(_DWORD *)(v2 + 452);
    }
    else
    {
      v11 = 1;
      while ( v9 != -4 )
      {
        v12 = v11 + 1;
        v7 = v5 & (v11 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v4 == *v8 )
          goto LABEL_3;
        v11 = v12;
      }
    }
  }
  sub_161E810(a1);
  return sub_164BE60(a1);
}
