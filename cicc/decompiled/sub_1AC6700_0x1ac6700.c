// Function: sub_1AC6700
// Address: 0x1ac6700
//
__int64 __fastcall sub_1AC6700(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int8 v9; // al
  __int64 v10; // r12
  unsigned __int8 v11; // dl
  __int64 result; // rax
  __int64 v13; // rdx
  int v14; // ecx
  __int64 v15; // r9
  int v16; // edi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r10
  __int64 v20; // rdx
  char v21; // al
  int v22; // edx
  int v23; // r11d

  v5 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = v5 - 24;
  v7 = v5 - 72;
  if ( (*a2 & 4) != 0 )
    v7 = v6;
  v8 = *(_QWORD *)v7;
  v9 = *(_BYTE *)(*(_QWORD *)v7 + 16LL);
  v10 = v8;
  v11 = v9;
  if ( v9 > 0x10u )
  {
    v13 = a1[6];
    if ( v13 == a1[7] )
      v13 = *(_QWORD *)(a1[9] - 8LL) + 512LL;
    v14 = *(_DWORD *)(v13 - 8);
    if ( !v14 )
      BUG();
    v15 = *(_QWORD *)(v13 - 24);
    v16 = v14 - 1;
    v17 = (v14 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v18 = (__int64 *)(v15 + 16LL * v17);
    v19 = *v18;
    if ( v8 != *v18 )
    {
      v22 = 1;
      while ( v19 != -8 )
      {
        v23 = v22 + 1;
        v17 = v16 & (v22 + v17);
        v18 = (__int64 *)(v15 + 16LL * v17);
        v19 = *v18;
        if ( v8 == *v18 )
          goto LABEL_15;
        v22 = v23;
      }
      goto LABEL_30;
    }
LABEL_15:
    v10 = v18[1];
    v11 = *(_BYTE *)(v10 + 16);
  }
  if ( !v11 )
    goto LABEL_5;
  if ( v11 == 1 )
  {
    v10 = *(_QWORD *)(v10 - 24);
    if ( v10 )
    {
      if ( !*(_BYTE *)(v10 + 16) )
      {
LABEL_5:
        if ( (unsigned __int8)sub_1AC6520(a1, a2, v10, a3) )
          return v10;
        return 0;
      }
      goto LABEL_8;
    }
LABEL_30:
    BUG();
  }
LABEL_8:
  if ( v9 != 5 || *(_WORD *)(v8 + 18) != 47 )
    return 0;
  v20 = *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
  v21 = *(_BYTE *)(v20 + 16);
  if ( v21 )
  {
    if ( v21 == 1 )
    {
      v20 = *(_QWORD *)(v20 - 24);
      if ( *(_BYTE *)(v20 + 16) )
        v20 = 0;
    }
    else
    {
      v20 = 0;
    }
  }
  if ( !(unsigned __int8)sub_1AC6520(a1, a2, v20, a3) )
    return 0;
  result = sub_14D66F0((__int64 *)v8, **(_QWORD **)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)), a1[80]);
  if ( *(_BYTE *)(result + 16) )
    return 0;
  return result;
}
