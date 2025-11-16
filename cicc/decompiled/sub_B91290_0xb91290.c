// Function: sub_B91290
// Address: 0xb91290
//
__int64 __fastcall sub_B91290(__int64 a1)
{
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rdi
  __int64 v5; // rsi
  int v6; // r8d
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  int v11; // edx
  int v12; // r10d

  v2 = ***(_QWORD ***)(a1 + 8);
  v3 = *(_DWORD *)(v2 + 624);
  v4 = *(_QWORD *)(v2 + 608);
  if ( v3 )
  {
    v5 = *(_QWORD *)(a1 + 24);
    v6 = v3 - 1;
    v7 = (v3 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v8 = (__int64 *)(v4 + 16LL * v7);
    v9 = *v8;
    if ( v5 == *v8 )
    {
LABEL_3:
      *v8 = -8192;
      --*(_DWORD *)(v2 + 616);
      ++*(_DWORD *)(v2 + 620);
    }
    else
    {
      v11 = 1;
      while ( v9 != -4096 )
      {
        v12 = v11 + 1;
        v7 = v6 & (v11 + v7);
        v8 = (__int64 *)(v4 + 16LL * v7);
        v9 = *v8;
        if ( v5 == *v8 )
          goto LABEL_3;
        v11 = v12;
      }
    }
  }
  sub_B91270(a1);
  return sub_BD7260(a1);
}
