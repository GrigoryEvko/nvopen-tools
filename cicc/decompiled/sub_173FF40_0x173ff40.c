// Function: sub_173FF40
// Address: 0x173ff40
//
__int64 __fastcall sub_173FF40(__int64 a1, int a2, int a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // edx
  __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // [rsp+Ch] [rbp-34h]

  v5 = *(_QWORD *)(a1 + 32);
  v6 = *(_QWORD *)(a1 + 40) + 40LL;
  if ( v6 == v5 )
    return 0;
  while ( 1 )
  {
    if ( !v5 )
      BUG();
    if ( *(_BYTE *)(v5 - 8) != 78 )
      return 0;
    v10 = *(_QWORD *)(v5 - 48);
    if ( *(_BYTE *)(v10 + 16) || (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
      return 0;
    v11 = *(_DWORD *)(v10 + 36);
    if ( (unsigned int)(v11 - 35) > 3 && a2 != v11 )
      break;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v6 == v5 )
      return 0;
  }
  if ( a3 != v11 )
    return 0;
  v21 = *(_DWORD *)(v5 - 4) & 0xFFFFFFF;
  if ( *(char *)(v5 - 1) >= 0 )
    goto LABEL_24;
  v12 = sub_1648A40(v5 - 24);
  v14 = v12 + v13;
  if ( *(char *)(v5 - 1) >= 0 )
  {
    if ( (unsigned int)(v14 >> 4) )
LABEL_28:
      BUG();
LABEL_24:
    v18 = 0;
    goto LABEL_16;
  }
  if ( !(unsigned int)((v14 - sub_1648A40(v5 - 24)) >> 4) )
    goto LABEL_24;
  if ( *(char *)(v5 - 1) >= 0 )
    goto LABEL_28;
  v15 = *(_DWORD *)(sub_1648A40(v5 - 24) + 8);
  if ( *(char *)(v5 - 1) >= 0 )
    BUG();
  v16 = sub_1648A40(v5 - 24);
  v18 = *(_DWORD *)(v16 + v17 - 4) - v15;
LABEL_16:
  if ( v21 - 1 != v18 )
  {
    v19 = 24LL * (unsigned int)(v21 - 1 - v18);
    v20 = 0;
    while ( *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) + v20) == *(_QWORD *)(v5
                                                                                           - 24LL
                                                                                           * (*(_DWORD *)(v5 - 4)
                                                                                            & 0xFFFFFFF)
                                                                                           + v20
                                                                                           - 24) )
    {
      v20 += 24;
      if ( v19 == v20 )
        goto LABEL_25;
    }
    return 0;
  }
LABEL_25:
  sub_170BC50(a4, v5 - 24);
  sub_170BC50(a4, a1);
  return 1;
}
