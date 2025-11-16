// Function: sub_1730740
// Address: 0x1730740
//
__int64 __fastcall sub_1730740(_QWORD *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // r14
  int v6; // eax
  __int64 v7; // rdx
  __int64 v11; // r10
  char v12; // cl
  unsigned __int64 v13; // r15
  __int64 **v14; // rbx
  __int64 v15; // rcx
  int v16; // esi
  __int64 v17; // rdi
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r10
  __int64 v22; // r11
  char v23; // al
  unsigned __int8 *v24; // rsi
  int v25; // eax
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // r9
  __int64 *v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // r9
  char v36; // al
  __int64 v37; // r13
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // [rsp+8h] [rbp-88h]
  __int64 v41; // [rsp+8h] [rbp-88h]
  int v42; // [rsp+10h] [rbp-80h]
  __int64 v43; // [rsp+10h] [rbp-80h]
  __int64 v44; // [rsp+10h] [rbp-80h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+18h] [rbp-78h]
  __int64 v47; // [rsp+18h] [rbp-78h]
  __int64 v48; // [rsp+20h] [rbp-70h]
  _QWORD *v49; // [rsp+20h] [rbp-70h]
  __int64 v50; // [rsp+20h] [rbp-70h]
  _QWORD *v51; // [rsp+20h] [rbp-70h]
  __int64 v52; // [rsp+20h] [rbp-70h]
  _QWORD *v53; // [rsp+20h] [rbp-70h]
  unsigned int v54; // [rsp+2Ch] [rbp-64h]
  _QWORD v55[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v56[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v57; // [rsp+50h] [rbp-40h]

  v5 = *(_QWORD *)(a2 - 48);
  v6 = *(unsigned __int8 *)(v5 + 16);
  if ( (unsigned __int8)v6 <= 0x17u )
    return 0;
  v7 = (unsigned __int8)v6;
  if ( (unsigned int)(unsigned __int8)v6 - 60 > 0xC )
    return 0;
  v11 = **(_QWORD **)(v5 - 24);
  v12 = *(_BYTE *)(v11 + 8);
  if ( v12 == 16 )
    v12 = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
  if ( v12 != 11 )
    return 0;
  v13 = *(_QWORD *)(a2 - 24);
  v54 = *(unsigned __int8 *)(a2 + 16) - 24;
  v14 = *(__int64 ***)a2;
  v15 = *(unsigned __int8 *)(v13 + 16);
  v16 = (unsigned __int8)v15;
  if ( (unsigned __int8)v15 <= 0x10u )
  {
    v17 = *(_QWORD *)(v5 + 8);
    if ( v17 )
    {
      v48 = a1[1];
      if ( *(_QWORD *)(v17 + 8) )
      {
LABEL_10:
        v16 = (unsigned __int8)v15;
        goto LABEL_11;
      }
      if ( (unsigned __int8)v6 != 61 )
        goto LABEL_23;
      if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
      {
        v26 = *(__int64 **)(v5 - 8);
      }
      else
      {
        v15 = 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
        v26 = (__int64 *)(v5 - v15);
      }
      v7 = *v26;
      v40 = v7;
      if ( !v7 )
        goto LABEL_23;
      v43 = **(_QWORD **)(v5 - 24);
      v46 = sub_15A43B0(v13, (__int64 **)v11, 0);
      v27 = sub_15A3CB0(v46, v14, 0);
      v15 = v46;
      v11 = v43;
      if ( v13 == v27 )
      {
        v57 = 257;
        v29 = sub_17066B0(v48, v54, v40, v46, v56, 0, a3, a4, a5);
        v57 = 257;
        v50 = v29;
        v30 = sub_1648A60(56, 1u);
        v11 = v43;
        if ( v30 )
        {
          v31 = v50;
          v51 = v30;
          sub_15FC690((__int64)v30, v31, (__int64)v14, (__int64)v56, 0);
          return (__int64)v51;
        }
      }
      else
      {
        v28 = *(_QWORD *)(v5 + 8);
        if ( v28 )
        {
          if ( *(_QWORD *)(v28 + 8) )
          {
            v15 = *(unsigned __int8 *)(v13 + 16);
            goto LABEL_10;
          }
          v6 = *(unsigned __int8 *)(v5 + 16);
          if ( (unsigned __int8)v6 <= 0x17u )
          {
            v25 = *(unsigned __int16 *)(v5 + 18);
            goto LABEL_24;
          }
LABEL_23:
          v25 = v6 - 24;
LABEL_24:
          if ( v25 == 38 )
          {
            if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
            {
              v32 = *(__int64 **)(v5 - 8);
            }
            else
            {
              v7 = 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
              v32 = (__int64 *)(v5 - v7);
            }
            v41 = *v32;
            if ( *v32 )
            {
              v44 = v11;
              v47 = sub_15A43B0(v13, (__int64 **)v11, 0);
              v33 = sub_15A4460(v47, v14, 0);
              v15 = v47;
              v11 = v44;
              if ( v13 == v33 )
              {
                v57 = 257;
                v52 = sub_17066B0(v48, v54, v41, v47, v56, 0, a3, a4, a5);
                v57 = 257;
                v34 = sub_1648A60(56, 1u);
                v11 = v44;
                if ( v34 )
                {
                  v35 = v52;
                  v53 = v34;
                  sub_15FC810((__int64)v34, v35, (__int64)v14, (__int64)v56, 0);
                  return (__int64)v53;
                }
              }
            }
          }
        }
      }
      v16 = *(unsigned __int8 *)(v13 + 16);
    }
  }
LABEL_11:
  if ( (unsigned __int8)(v16 - 60) > 0xCu )
    return 0;
  if ( *(_BYTE *)(v5 + 16) != (_BYTE)v16 )
    return 0;
  v49 = *(_QWORD **)(v13 - 24);
  if ( v11 != *v49 )
    return 0;
  v42 = v16 - 24;
  v45 = *(_QWORD *)(v5 - 24);
  v18 = sub_17290A0((__int64)a1, (_QWORD *)v5, v7, v15);
  v21 = v45;
  v22 = (__int64)v49;
  if ( !v18 || (v36 = sub_17290A0((__int64)a1, (_QWORD *)v13, v19, v20), v22 = (__int64)v49, v21 = v45, !v36) )
  {
    if ( v54 != 28 )
    {
      v23 = *(_BYTE *)(v21 + 16);
      if ( v23 == 75 )
      {
        if ( *(_BYTE *)(v22 + 16) == 75 )
        {
          v24 = v54 == 26 ? sub_17306D0(a1, v21, v22, a2, a3, a4, a5) : sub_172F880(a1, v21, v22, a2, a3, a4, a5);
          if ( v24 )
            goto LABEL_21;
        }
      }
      else if ( v23 == 76 && *(_BYTE *)(v22 + 16) == 76 )
      {
        v24 = sub_1729330((__int64)a1, v21, v22, v54 == 26);
        if ( v24 )
        {
LABEL_21:
          v57 = 257;
          return sub_15FDBD0(v42, (__int64)v24, (__int64)v14, (__int64)v56, 0);
        }
      }
    }
    return 0;
  }
  v37 = a1[1];
  v55[0] = sub_1649960(a2);
  v55[1] = v38;
  v57 = 261;
  v56[0] = (__int64)v55;
  v39 = sub_17066B0(v37, v54, v45, (__int64)v49, v56, 0, a3, a4, a5);
  v57 = 257;
  return sub_15FDBD0(v42, v39, (__int64)v14, (__int64)v56, 0);
}
