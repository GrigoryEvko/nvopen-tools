// Function: sub_38ECF20
// Address: 0x38ecf20
//
__int64 __fastcall sub_38ECF20(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // r14
  _DWORD *v4; // rax
  unsigned int v5; // r13d
  __int64 v7; // rax
  __int64 v8; // r8
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  _QWORD *v12; // r11
  __int64 v13; // r14
  unsigned int v14; // r12d
  int v15; // r15d
  __int64 v16; // r9
  unsigned int v17; // r13d
  char v18; // al
  int v19; // eax
  unsigned int v20; // edx
  unsigned int v21; // r13d
  unsigned int v22; // eax
  unsigned int v23; // r12d
  unsigned int v24; // eax
  __int64 v25; // r14
  _QWORD *v26; // r15
  const char *v27; // rax
  size_t v28; // rsi
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rax
  size_t v33; // rsi
  char v34; // r12
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // r13
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rcx
  unsigned __int64 v42; // r13
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rcx
  unsigned __int64 v46; // r13
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rcx
  unsigned __int64 v50; // r13
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rcx
  unsigned __int64 v54; // r13
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rcx
  unsigned __int64 v58; // r13
  unsigned __int64 v59; // rdx
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rcx
  unsigned __int64 v62; // r13
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rcx
  _QWORD *v66; // [rsp+8h] [rbp-78h]
  _QWORD *v67; // [rsp+8h] [rbp-78h]
  __int64 v68; // [rsp+10h] [rbp-70h]
  __int64 v69; // [rsp+10h] [rbp-70h]
  _QWORD *v70; // [rsp+10h] [rbp-70h]
  _QWORD *v71; // [rsp+10h] [rbp-70h]
  _QWORD *v72; // [rsp+10h] [rbp-70h]
  _QWORD *v73; // [rsp+10h] [rbp-70h]
  _QWORD *v74; // [rsp+10h] [rbp-70h]
  _QWORD *v75; // [rsp+10h] [rbp-70h]
  _QWORD *v76; // [rsp+10h] [rbp-70h]
  char v77; // [rsp+18h] [rbp-68h]
  __int64 v78; // [rsp+18h] [rbp-68h]
  __int64 v79; // [rsp+18h] [rbp-68h]
  __int64 v80; // [rsp+18h] [rbp-68h]
  __int64 v81; // [rsp+18h] [rbp-68h]
  __int64 v82; // [rsp+18h] [rbp-68h]
  __int64 v83; // [rsp+18h] [rbp-68h]
  __int64 v84; // [rsp+18h] [rbp-68h]
  _QWORD v85[2]; // [rsp+30h] [rbp-50h] BYREF
  char v86; // [rsp+40h] [rbp-40h]
  char v87; // [rsp+41h] [rbp-3Fh]

  v2 = a1;
  v87 = 1;
  v85[0] = "expected string";
  v86 = 3;
  v4 = (_DWORD *)sub_3909460(a1);
  v5 = sub_3909CB0(a1, *v4 != 3, v85);
  if ( (_BYTE)v5 )
    return v5;
  sub_2241130(a2, 0, a2[1], byte_3F871B3, 0);
  v7 = sub_3909460(a1);
  v9 = *(_QWORD *)(v7 + 16);
  v10 = v7;
  v11 = v9 - 1;
  if ( !v9 )
    goto LABEL_29;
  if ( v9 == 1 )
    v11 = 1;
  if ( v11 > v9 )
    LODWORD(v11) = v9;
  if ( (_DWORD)v11 == 1 )
  {
LABEL_29:
    sub_38EB180(v2);
    return v5;
  }
  v12 = v85;
  v13 = *(_QWORD *)(v10 + 8) + 1LL;
  v14 = 0;
  v15 = v11 - 1;
  v16 = a1;
  while ( 1 )
  {
    v17 = v14 + 1;
    v18 = *(_BYTE *)(v13 + v14);
    if ( v18 != 92 )
    {
      v28 = a2[1];
      v29 = *a2;
      v30 = v28 + 1;
      if ( (unsigned __int64 *)*a2 == a2 + 2 )
        v31 = 15;
      else
        v31 = a2[2];
      if ( v30 > v31 )
      {
        v66 = v12;
        v68 = v16;
        v77 = v18;
        sub_2240BB0(a2, v28, 0, 0, 1u);
        v29 = *a2;
        v12 = v66;
        v16 = v68;
        v18 = v77;
      }
      *(_BYTE *)(v29 + v28) = v18;
      v32 = *a2;
      a2[1] = v30;
      v14 = v17;
      *(_BYTE *)(v32 + v28 + 1) = 0;
      goto LABEL_27;
    }
    if ( v15 == v17 )
      break;
    v19 = *(char *)(v13 + v17);
    v20 = v19 - 48;
    if ( (unsigned int)(v19 - 48) > 7 )
    {
      if ( (_BYTE)v19 != 34 )
      {
        switch ( (char)v19 )
        {
          case '\\':
            v54 = a2[1];
            v55 = *a2;
            v56 = v54 + 1;
            if ( (unsigned __int64 *)*a2 == a2 + 2 )
              v57 = 15;
            else
              v57 = a2[2];
            if ( v56 > v57 )
            {
              v71 = v12;
              v79 = v16;
              sub_2240BB0(a2, a2[1], 0, 0, 1u);
              v55 = *a2;
              v12 = v71;
              v16 = v79;
              v56 = v54 + 1;
            }
            *(_BYTE *)(v55 + v54) = 92;
            v14 += 2;
            a2[1] = v56;
            *(_BYTE *)(*a2 + v54 + 1) = 0;
            goto LABEL_27;
          case 'b':
            v50 = a2[1];
            v51 = *a2;
            v52 = v50 + 1;
            if ( (unsigned __int64 *)*a2 == a2 + 2 )
              v53 = 15;
            else
              v53 = a2[2];
            if ( v52 > v53 )
            {
              v72 = v12;
              v80 = v16;
              sub_2240BB0(a2, a2[1], 0, 0, 1u);
              v51 = *a2;
              v12 = v72;
              v16 = v80;
              v52 = v50 + 1;
            }
            *(_BYTE *)(v51 + v50) = 8;
            v14 += 2;
            a2[1] = v52;
            *(_BYTE *)(*a2 + v50 + 1) = 0;
            goto LABEL_27;
          case 'f':
            v46 = a2[1];
            v47 = *a2;
            v48 = v46 + 1;
            if ( (unsigned __int64 *)*a2 == a2 + 2 )
              v49 = 15;
            else
              v49 = a2[2];
            if ( v48 > v49 )
            {
              v73 = v12;
              v81 = v16;
              sub_2240BB0(a2, a2[1], 0, 0, 1u);
              v47 = *a2;
              v12 = v73;
              v16 = v81;
              v48 = v46 + 1;
            }
            *(_BYTE *)(v47 + v46) = 12;
            v14 += 2;
            a2[1] = v48;
            *(_BYTE *)(*a2 + v46 + 1) = 0;
            goto LABEL_27;
          case 'n':
            v42 = a2[1];
            v43 = *a2;
            v44 = v42 + 1;
            if ( (unsigned __int64 *)*a2 == a2 + 2 )
              v45 = 15;
            else
              v45 = a2[2];
            if ( v44 > v45 )
            {
              v74 = v12;
              v82 = v16;
              sub_2240BB0(a2, a2[1], 0, 0, 1u);
              v43 = *a2;
              v12 = v74;
              v16 = v82;
              v44 = v42 + 1;
            }
            *(_BYTE *)(v43 + v42) = 10;
            v14 += 2;
            a2[1] = v44;
            *(_BYTE *)(*a2 + v42 + 1) = 0;
            goto LABEL_27;
          case 'r':
            v58 = a2[1];
            v59 = *a2;
            v60 = v58 + 1;
            if ( (unsigned __int64 *)*a2 == a2 + 2 )
              v61 = 15;
            else
              v61 = a2[2];
            if ( v60 > v61 )
            {
              v70 = v12;
              v78 = v16;
              sub_2240BB0(a2, a2[1], 0, 0, 1u);
              v59 = *a2;
              v12 = v70;
              v16 = v78;
              v60 = v58 + 1;
            }
            *(_BYTE *)(v59 + v58) = 13;
            v14 += 2;
            a2[1] = v60;
            *(_BYTE *)(*a2 + v58 + 1) = 0;
            goto LABEL_27;
          case 't':
            v38 = a2[1];
            v39 = *a2;
            v40 = v38 + 1;
            if ( (unsigned __int64 *)*a2 == a2 + 2 )
              v41 = 15;
            else
              v41 = a2[2];
            if ( v40 > v41 )
            {
              v75 = v12;
              v83 = v16;
              sub_2240BB0(a2, a2[1], 0, 0, 1u);
              v39 = *a2;
              v12 = v75;
              v16 = v83;
              v40 = v38 + 1;
            }
            *(_BYTE *)(v39 + v38) = 9;
            v14 += 2;
            a2[1] = v40;
            *(_BYTE *)(*a2 + v38 + 1) = 0;
            goto LABEL_27;
          default:
            v87 = 1;
            v25 = v16;
            v26 = v12;
            v27 = "invalid escape sequence (unrecognized character)";
            goto LABEL_31;
        }
      }
      v62 = a2[1];
      v63 = *a2;
      v64 = v62 + 1;
      if ( (unsigned __int64 *)*a2 == a2 + 2 )
        v65 = 15;
      else
        v65 = a2[2];
      if ( v64 > v65 )
      {
        v76 = v12;
        v84 = v16;
        sub_2240BB0(a2, a2[1], 0, 0, 1u);
        v63 = *a2;
        v12 = v76;
        v16 = v84;
        v64 = v62 + 1;
      }
      *(_BYTE *)(v63 + v62) = 34;
      v14 += 2;
      a2[1] = v64;
      *(_BYTE *)(*a2 + v62 + 1) = 0;
    }
    else
    {
      v21 = v14 + 2;
      if ( v14 + 2 != v15 )
      {
        v22 = *(char *)(v13 + v21) - 48;
        if ( v22 <= 7 )
        {
          v23 = v14 + 3;
          v20 = v22 + 8 * v20;
          if ( v23 != v15 )
          {
            v24 = *(char *)(v13 + v23) - 48;
            if ( v24 <= 7 )
            {
              v20 = v24 + 8 * v20;
              v21 = v23;
            }
          }
          if ( v20 > 0xFF )
          {
            v87 = 1;
            v25 = v16;
            v26 = v12;
            v27 = "invalid octal escape sequence (out of range)";
            goto LABEL_31;
          }
          ++v21;
        }
      }
      v33 = a2[1];
      v34 = v20;
      v35 = *a2;
      v36 = v33 + 1;
      if ( (unsigned __int64 *)*a2 == a2 + 2 )
        v37 = 15;
      else
        v37 = a2[2];
      if ( v36 > v37 )
      {
        v67 = v12;
        v69 = v16;
        sub_2240BB0(a2, v33, 0, 0, 1u);
        v35 = *a2;
        v12 = v67;
        v16 = v69;
        v36 = v33 + 1;
      }
      *(_BYTE *)(v35 + v33) = v34;
      v14 = v21;
      a2[1] = v36;
      *(_BYTE *)(*a2 + v33 + 1) = 0;
    }
LABEL_27:
    if ( v15 == v14 )
    {
      v5 = 0;
      v2 = v16;
      goto LABEL_29;
    }
  }
  v87 = 1;
  v25 = v16;
  v26 = v12;
  v27 = "unexpected backslash at end of string";
LABEL_31:
  v85[0] = v27;
  v86 = 3;
  return (unsigned int)sub_3909CF0(v25, v26, 0, 0, v8, v16);
}
