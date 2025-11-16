// Function: sub_13D3F90
// Address: 0x13d3f90
//
__int64 __fastcall sub_13D3F90(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // rdi
  unsigned __int8 v4; // al
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r12
  char v7; // dl
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  int v11; // r14d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  int v18; // r14d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 *v22; // rcx
  __int64 *v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // rsi
  unsigned __int8 v26; // al
  __int64 v27; // rdi

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = *(_BYTE *)(v2 + 16);
  if ( v4 <= 0x17u )
    goto LABEL_4;
  v5 = v2;
  v6 = v2;
  v7 = v2 | 4;
  if ( v4 != 78 )
  {
    if ( v4 != 29 )
    {
LABEL_4:
      v6 = 0;
      v5 = 0;
      goto LABEL_5;
    }
    v7 = v2;
  }
  if ( (v7 & 4) == 0 )
  {
LABEL_5:
    if ( *(char *)(v6 + 23) < 0 )
    {
      v8 = sub_1648A40(v6);
      v10 = v8 + v9;
      if ( *(char *)(v6 + 23) >= 0 )
      {
        if ( (unsigned int)(v10 >> 4) )
          goto LABEL_35;
      }
      else if ( (unsigned int)((v10 - sub_1648A40(v6)) >> 4) )
      {
        if ( *(char *)(v6 + 23) < 0 )
        {
          v11 = *(_DWORD *)(sub_1648A40(v6) + 8);
          if ( *(char *)(v6 + 23) >= 0 )
            BUG();
          v12 = sub_1648A40(v6);
          v14 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
          goto LABEL_21;
        }
LABEL_35:
        BUG();
      }
    }
    v14 = -72;
LABEL_21:
    v22 = (__int64 *)(v6 + v14);
    v23 = (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    v24 = (__int64 *)(v6 - 72);
    goto LABEL_22;
  }
  if ( *(char *)(v2 + 23) < 0 )
  {
    v15 = sub_1648A40(v2);
    v17 = v15 + v16;
    if ( *(char *)(v2 + 23) >= 0 )
    {
      if ( (unsigned int)(v17 >> 4) )
        goto LABEL_33;
    }
    else if ( (unsigned int)((v17 - sub_1648A40(v2)) >> 4) )
    {
      if ( *(char *)(v2 + 23) < 0 )
      {
        v18 = *(_DWORD *)(sub_1648A40(v2) + 8);
        if ( *(char *)(v2 + 23) >= 0 )
          BUG();
        v19 = sub_1648A40(v2);
        v21 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v18);
        goto LABEL_29;
      }
LABEL_33:
      BUG();
    }
  }
  v21 = -24;
LABEL_29:
  v22 = (__int64 *)(v2 + v21);
  v23 = (__int64 *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
  v24 = (__int64 *)(v2 - 24);
LABEL_22:
  v25 = *v24;
  v26 = *(_BYTE *)(v6 + 16);
  v27 = 0;
  if ( v26 <= 0x17u )
    return sub_13D3B30(v27, v25, v23, v22, a2);
  if ( v26 != 78 )
  {
    if ( v26 == 29 )
      v27 = v5;
    return sub_13D3B30(v27, v25, v23, v22, a2);
  }
  return sub_13D3B30(v5 | 4, v25, v23, v22, a2);
}
