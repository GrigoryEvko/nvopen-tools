// Function: sub_2D61230
// Address: 0x2d61230
//
__int64 __fastcall sub_2D61230(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r8
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 v7; // rdx
  __int64 v8; // r10
  __int64 v9; // rcx
  int v11; // edx
  int v12; // r11d

  v2 = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)v2 )
  {
LABEL_7:
    v9 = *(_QWORD *)(a1 + 32);
    return v9 + 1064LL * *(unsigned int *)(a1 + 40);
  }
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v2 - 1) & (((unsigned int)*(_QWORD *)(a2 + 16) >> 9) ^ ((unsigned int)v4 >> 4));
  v7 = v5 + 32LL * v6;
  v8 = *(_QWORD *)(v7 + 16);
  if ( v4 != v8 )
  {
    v11 = 1;
    while ( v8 != -4096 )
    {
      v12 = v11 + 1;
      v6 = (v2 - 1) & (v11 + v6);
      v7 = v5 + 32LL * v6;
      v8 = *(_QWORD *)(v7 + 16);
      if ( v4 == v8 )
        goto LABEL_3;
      v11 = v12;
    }
    goto LABEL_7;
  }
LABEL_3:
  v9 = *(_QWORD *)(a1 + 32);
  if ( v7 != v5 + 32 * v2 )
    return v9 + 1064LL * *(unsigned int *)(v7 + 24);
  return v9 + 1064LL * *(unsigned int *)(a1 + 40);
}
