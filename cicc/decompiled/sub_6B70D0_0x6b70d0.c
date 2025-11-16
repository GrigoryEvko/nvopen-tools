// Function: sub_6B70D0
// Address: 0x6b70d0
//
__int64 __fastcall sub_6B70D0(_BYTE *a1, __int64 a2, __int64 a3)
{
  _BOOL4 v5; // r13d
  _BYTE *v6; // rbx
  __int64 *v7; // rsi
  __int64 v8; // rdi
  void *v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  int v13; // r15d
  int v14; // edx
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // rax
  __int64 v18; // r10
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // edx
  __int64 v24; // rax
  __int64 *v25; // r13
  char v26; // al
  _BOOL4 v27; // r13d
  int v28; // eax
  unsigned int v29; // r10d
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rax
  __int64 i; // rdx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // r8
  int v37; // eax
  int v38; // eax
  __int64 v39; // rdx
  __int64 v40; // r10
  int v41; // eax
  int v42; // eax
  _QWORD *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // rcx
  int v48; // eax
  __int64 v49; // [rsp+10h] [rbp-330h]
  int v50; // [rsp+10h] [rbp-330h]
  __int64 v51; // [rsp+18h] [rbp-328h]
  __int64 v52; // [rsp+18h] [rbp-328h]
  __int64 v53; // [rsp+18h] [rbp-328h]
  __int64 v54; // [rsp+20h] [rbp-320h]
  _QWORD *v55; // [rsp+20h] [rbp-320h]
  __int64 v56; // [rsp+20h] [rbp-320h]
  __int64 v57; // [rsp+20h] [rbp-320h]
  __int64 v58; // [rsp+20h] [rbp-320h]
  int v59; // [rsp+28h] [rbp-318h]
  __int64 v60; // [rsp+28h] [rbp-318h]
  unsigned int v61; // [rsp+38h] [rbp-308h] BYREF
  int v62; // [rsp+3Ch] [rbp-304h] BYREF
  __int64 v63; // [rsp+40h] [rbp-300h] BYREF
  __int64 v64; // [rsp+48h] [rbp-2F8h] BYREF
  _BYTE v65[352]; // [rsp+50h] [rbp-2F0h] BYREF
  _DWORD v66[4]; // [rsp+1B0h] [rbp-190h] BYREF
  char v67; // [rsp+1C0h] [rbp-180h]
  char v68; // [rsp+1C1h] [rbp-17Fh]
  __int64 v69; // [rsp+1F4h] [rbp-14Ch]
  __int64 v70; // [rsp+1FCh] [rbp-144h]
  __int64 v71; // [rsp+208h] [rbp-138h]
  __int64 *v72; // [rsp+240h] [rbp-100h]

  v62 = 0;
  v5 = (*(_BYTE *)(qword_4D03C50 + 20LL) & 8) != 0;
  if ( a2 )
  {
    v6 = v65;
    sub_6F8AB0(a2, (unsigned int)v65, (unsigned int)v66, 0, (unsigned int)&v64, (unsigned int)&v61, 0);
  }
  else
  {
    v6 = a1;
    v64 = *(_QWORD *)&dword_4F063F8;
    v61 = dword_4F06650[0];
  }
  v7 = 0;
  sub_6E16F0(*((_QWORD *)v6 + 11), 0);
  v8 = (__int64)v6;
  sub_6E17F0(v6);
  v12 = qword_4D03C50;
  if ( v5 && v6[16] == 1 )
  {
    v18 = *((_QWORD *)v6 + 18);
    v11 = (__int64)(v6 + 68);
    v9 = (void *)(qword_4D03C50 + 104LL);
    if ( *(_BYTE *)(v18 + 24) == 1 )
    {
      while ( 1 )
      {
        v19 = *(_BYTE *)(v18 + 56);
        if ( v19 != 91 )
          break;
        v18 = *(_QWORD *)(*(_QWORD *)(v18 + 72) + 16LL);
        if ( *(_BYTE *)(v18 + 24) != 1 )
          goto LABEL_4;
        v11 = qword_4D03C50 + 104LL;
      }
      if ( (unsigned __int8)(v19 - 105) <= 4u )
      {
        v8 = v18;
        v54 = v11;
        v60 = v18;
        v20 = sub_6EE7B0(v18);
        v11 = v54;
        if ( *(_BYTE *)(v20 + 140) != 7
          || (v51 = v54,
              v8 = *(_QWORD *)(v20 + 160),
              v56 = v20,
              v38 = sub_8D2600(v8),
              v39 = v56,
              v11 = v51,
              v40 = v60,
              v38) )
        {
          v12 = qword_4D03C50;
        }
        else
        {
          v57 = *(_QWORD *)(v56 + 168);
          if ( dword_4F077C4 == 2 )
          {
            v48 = sub_8D23B0(v8);
            v11 = v51;
            v40 = v60;
            if ( v48 )
            {
              sub_8AE000(v8);
              v40 = v60;
              v11 = v51;
            }
          }
          if ( (*(_BYTE *)(v57 + 17) & 8) == 0 )
          {
            v49 = v40;
            v52 = v11;
            v41 = sub_8D23B0(v8);
            v11 = v52;
            v40 = v49;
            if ( v41 )
            {
              v42 = sub_6E5430(v8, 0, v39, v9, v10, v52);
              v11 = v52;
              v40 = v49;
              if ( v42 )
              {
                v43 = (_QWORD *)sub_72B0F0(*(_QWORD *)(v49 + 72), 0);
                v7 = (__int64 *)v52;
                sub_625A80(v8, v52, v43, v44, v45);
                v40 = v49;
                v11 = v52;
              }
            }
          }
          *(_BYTE *)(v57 + 17) |= 8u;
          if ( (*(_BYTE *)(v57 + 16) & 0x20) != 0 )
          {
            v50 = v11;
            v53 = v40;
            v58 = sub_730FF0(v40, v7, v39);
            v46 = sub_6ECAE0(v8, 0, 0, 1, 4, v50, (__int64)&v63);
            v47 = 22;
            v7 = (__int64 *)v46;
            v8 = v53;
            while ( v47 )
            {
              *(_DWORD *)v8 = *(_DWORD *)v7;
              v7 = (__int64 *)((char *)v7 + 4);
              v8 += 4;
              --v47;
            }
            v9 = (void *)v58;
            *(_QWORD *)(v63 + 56) = v58;
            v12 = qword_4D03C50;
          }
          else
          {
            v12 = qword_4D03C50;
          }
        }
      }
    }
  }
LABEL_4:
  if ( dword_4F077C4 != 2 )
  {
    v9 = &unk_4F07778;
    if ( unk_4F07778 > 199900 && (*(_BYTE *)(v12 + 17) & 1) == 0 )
    {
      v59 = 1;
      if ( a2 )
        goto LABEL_9;
LABEL_16:
      sub_7B8B50(v8, v7, v12, v9);
      v7 = 0;
      v8 = (__int64)v66;
      sub_69ED20((__int64)v66, 0, 1, 0);
      if ( dword_4F077C4 != 2 )
        goto LABEL_9;
      goto LABEL_45;
    }
  }
  if ( (*(_BYTE *)(v12 + 19) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(v8, v7, v12, v9, v10, v11) )
    {
      v7 = &v64;
      v8 = 57;
      sub_6851C0(0x39u, &v64);
    }
    if ( !a2 )
    {
      sub_7B8B50(v8, v7, v21, v22);
      sub_69ED20((__int64)v66, 0, 1, 0);
    }
    sub_6E6260(a3);
    sub_6E6450(v6);
    sub_6E6450(v66);
LABEL_43:
    v23 = *((_DWORD *)v6 + 17);
    *(_WORD *)(a3 + 72) = *((_WORD *)v6 + 36);
    *(_DWORD *)(a3 + 68) = v23;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(a3 + 68);
    v24 = v70;
    *(_QWORD *)(a3 + 76) = v70;
    unk_4F061D8 = v24;
    sub_6E3280(a3, &v64);
    sub_6E3BA0(a3, &v64, v61, 0);
    return sub_6E26D0(2, a3);
  }
  v59 = 0;
  if ( !a2 )
    goto LABEL_16;
  if ( dword_4F077C4 != 2 )
    goto LABEL_9;
LABEL_45:
  if ( (unsigned int)sub_68FE10(v6, 1, 1) || (v7 = 0, v8 = (__int64)v66, (unsigned int)sub_68FE10(v66, 0, 1)) )
  {
    v7 = 0;
    v8 = 39;
    sub_84EC30(39, 0, 0, 0, 1, (_DWORD)v6, (__int64)v66, (__int64)&v64, v61, 0, 0, a3, 0, 0, (__int64)&v62);
    v13 = v62;
    if ( v62 )
      goto LABEL_10;
    goto LABEL_18;
  }
LABEL_9:
  v13 = v62;
  if ( v62 )
  {
LABEL_10:
    if ( !v5 )
      goto LABEL_12;
    if ( v67 != 1 )
      goto LABEL_12;
    v25 = v72;
    if ( *((_BYTE *)v72 + 24) != 1 )
      goto LABEL_12;
    while ( 1 )
    {
      v26 = *((_BYTE *)v25 + 56);
      if ( v26 == 91 )
      {
        v25 = *(__int64 **)(v25[9] + 16);
      }
      else if ( v26 != 25 )
      {
        if ( (unsigned __int8)(v26 - 105) <= 4u )
        {
          v32 = *v25;
          for ( i = *(unsigned __int8 *)(*v25 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v32 + 140) )
            v32 = *(_QWORD *)(v32 + 160);
          if ( (*(_BYTE *)(v32 + 141) & 0x20) != 0 && (_BYTE)i != 1 )
          {
            if ( (unsigned int)sub_6E5430(v8, v7, i, v9, v10, v11) )
            {
              v55 = (_QWORD *)sub_72B0F0(v25[9], 0);
              v34 = sub_6EE7B0(v25);
              sub_625A80(*(_QWORD *)(v34 + 160), (__int64)v25 + 28, v55, v35, v36);
            }
          }
        }
        goto LABEL_12;
      }
      if ( *((_BYTE *)v25 + 24) != 1 )
        goto LABEL_12;
    }
  }
LABEL_18:
  sub_68CD10((__int64 *)v6, 0);
  if ( v5 )
  {
    v17 = qword_4D03C50;
    if ( v67 == 1 && *((_BYTE *)v72 + 24) == 1 && (unsigned __int8)(*((_BYTE *)v72 + 56) - 105) <= 4u )
      *(_QWORD *)(qword_4D03C50 + 104LL) = v69;
    *(_BYTE *)(v17 + 20) |= 8u;
  }
  if ( dword_4F077C4 == 2 )
  {
    v27 = 1;
    sub_6F6C80(v66);
    v28 = sub_6ED0A0(v66);
    v29 = (unsigned int)v66;
    v13 = v28;
    if ( !v28 )
    {
      v37 = sub_6ED1E0(v66);
      v29 = (unsigned int)v66;
      v27 = v37 != 0;
    }
  }
  else
  {
    if ( !dword_4F077C0 || qword_4F077A8 > 0x9C3Fu )
    {
      sub_6F69D0(v66, 0);
      sub_700E50(91, (_DWORD)v6, (unsigned int)v66, v66[0], 0, a3, (__int64)&v64, v61, 0);
      if ( word_4D04898 || *(_BYTE *)(a3 + 16) != 2 || v59 )
        goto LABEL_12;
      *(_BYTE *)(a3 + 313) |= 4u;
      goto LABEL_43;
    }
    v27 = 0;
    sub_6F69D0(v66, 4);
    v29 = (unsigned int)v66;
    if ( v68 == 1 )
    {
      v31 = sub_6ED0A0(v66);
      v29 = (unsigned int)v66;
      v27 = v31 == 0;
    }
  }
  sub_700E50(91, (_DWORD)v6, v29, v66[0], v27, a3, (__int64)&v64, v61, 0);
  if ( !word_4D04898 && *(_BYTE *)(a3 + 16) == 2 && !v59 )
    *(_BYTE *)(a3 + 313) |= 4u;
  if ( v27 )
  {
    v30 = v71;
    v71 = 0;
    *(_QWORD *)(a3 + 88) = v30;
    if ( v13 )
      sub_6ED1A0(a3);
  }
LABEL_12:
  v14 = *((_DWORD *)v6 + 17);
  *(_WORD *)(a3 + 72) = *((_WORD *)v6 + 36);
  *(_DWORD *)(a3 + 68) = v14;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a3 + 68);
  v15 = v70;
  *(_QWORD *)(a3 + 76) = v70;
  unk_4F061D8 = v15;
  sub_6E3280(a3, &v64);
  result = sub_6E3BA0(a3, &v64, v61, 0);
  if ( !v59 )
    return sub_6E26D0(2, a3);
  return result;
}
