// Function: sub_2E85750
// Address: 0x2e85750
//
bool __fastcall sub_2E85750(__int64 a1, __int64 *a2, char a3, __int64 *a4, __int64 *a5)
{
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r10
  __int64 v15; // rdi
  __int64 v16; // rbx
  bool v17; // r8
  bool v18; // al
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // r15
  bool v21; // al
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // r11
  char v24; // al
  bool v25; // si
  bool v26; // r15
  unsigned __int64 v27; // rsi
  char v28; // al
  unsigned __int64 v29; // r14
  unsigned __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v41; // rdi
  __int64 v42; // rax
  char v43; // al
  unsigned __int64 v44; // rsi
  char v45; // r8
  int v46; // edi
  unsigned __int64 v47; // rcx
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rcx
  unsigned __int64 v50; // rsi
  char v51; // r8
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // r10
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // r10
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rcx
  unsigned __int64 v58; // rsi
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // [rsp+0h] [rbp-100h]
  unsigned __int64 v61; // [rsp+8h] [rbp-F8h]
  __int64 v62; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v63; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v64; // [rsp+18h] [rbp-E8h]
  bool v65; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v66; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v67; // [rsp+20h] [rbp-E0h]
  __int64 v68; // [rsp+28h] [rbp-D8h]
  __int64 v69; // [rsp+28h] [rbp-D8h]
  __int64 v70; // [rsp+28h] [rbp-D8h]
  __int64 v71; // [rsp+30h] [rbp-D0h]
  __int64 v72; // [rsp+38h] [rbp-C8h]
  __int64 v73; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v74; // [rsp+48h] [rbp-B8h]
  __int64 v76; // [rsp+58h] [rbp-A8h]
  unsigned __int64 v77; // [rsp+60h] [rbp-A0h]
  char v79; // [rsp+6Dh] [rbp-93h]
  char v80; // [rsp+6Eh] [rbp-92h]
  bool v81; // [rsp+6Fh] [rbp-91h]
  _QWORD v82[6]; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int64 v83; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 v84; // [rsp+A8h] [rbp-58h]
  __int64 v85; // [rsp+B0h] [rbp-50h]
  __int64 v86; // [rsp+B8h] [rbp-48h]
  __int64 v87; // [rsp+C0h] [rbp-40h]
  __int64 v88; // [rsp+C8h] [rbp-38h]

  v8 = a5[1];
  v9 = a4[1];
  v10 = v8;
  v76 = 0x4000000000000000LL;
  v80 = 0;
  if ( v9 <= v8 )
    v10 = a4[1];
  v72 = v10;
  v11 = a4[3];
  v12 = -1;
  if ( (v11 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    v44 = v11 >> 3;
    v45 = a4[3] & 2;
    if ( (a4[3] & 6) == 2 || (a4[3] & 1) != 0 )
    {
      v76 = 0;
      v57 = HIWORD(v11);
      if ( !v45 )
        v57 = HIDWORD(v11);
      v12 = (v57 + 7) >> 3;
    }
    else
    {
      v46 = (unsigned __int16)((unsigned int)v11 >> 8);
      v47 = v11;
      v48 = HIDWORD(v11);
      v49 = HIWORD(v47);
      if ( !v45 )
        LODWORD(v49) = v48;
      v12 = ((unsigned __int64)(unsigned int)(v46 * v49) + 7) >> 3;
      if ( (v44 & 1) != 0 )
        v12 |= 0x4000000000000000uLL;
      else
        v76 = 0;
    }
    v13 = a5[3];
    v80 = 1;
    if ( (v13 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
      goto LABEL_5;
  }
  else
  {
    v13 = a5[3];
    if ( (v13 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
    {
LABEL_5:
      v79 = 0;
      v14 = -1;
      v71 = 0x4000000000000000LL;
      v73 = 0x4000000000000000LL;
      goto LABEL_6;
    }
  }
  v50 = v13 >> 3;
  v51 = a5[3] & 2;
  if ( (a5[3] & 6) == 2 || (a5[3] & 1) != 0 )
  {
    v71 = 0;
    v58 = HIDWORD(v13);
    v59 = HIWORD(v13);
    if ( v51 )
      v58 = v59;
    v14 = (v58 + 7) >> 3;
  }
  else
  {
    v52 = v13;
    v53 = v13;
    v54 = HIWORD(v13);
    v55 = HIDWORD(v53);
    v56 = v52 >> 8;
    if ( !v51 )
      LODWORD(v54) = v55;
    v14 = ((unsigned __int64)((unsigned int)v54 * (unsigned __int16)v56) + 7) >> 3;
    if ( (v50 & 1) != 0 )
    {
      v71 = 0x4000000000000000LL;
      v14 |= 0x4000000000000000uLL;
    }
    else
    {
      v71 = 0;
    }
  }
  v79 = 1;
  v73 = (v14 | v12) & 0x4000000000000000LL;
LABEL_6:
  v15 = *a4;
  if ( *a4 )
  {
    if ( ((v15 >> 2) & 1) == 0 )
    {
      v16 = *a5;
      v77 = v15 & 0xFFFFFFFFFFFFFFF8LL;
      v17 = (v15 & 0xFFFFFFFFFFFFFFF8LL) != 0;
      if ( *a5 )
        goto LABEL_9;
LABEL_58:
      v74 = 0;
      v18 = 0;
      if ( !v15 )
        goto LABEL_23;
      if ( (v15 & 4) != 0 )
        goto LABEL_14;
LABEL_42:
      if ( !v16 )
        goto LABEL_23;
      goto LABEL_43;
    }
    v16 = *a5;
    v17 = 0;
    v77 = 0;
    if ( !*a5 )
    {
      v74 = 0;
      v18 = 0;
      if ( ((v15 >> 2) & 1) == 0 )
        goto LABEL_23;
      goto LABEL_14;
    }
  }
  else
  {
    v16 = *a5;
    v17 = 0;
    v77 = 0;
    if ( !*a5 )
      goto LABEL_58;
  }
LABEL_9:
  if ( (v16 & 4) == 0 )
  {
    v74 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    v18 = (v16 & 0xFFFFFFFFFFFFFFF8LL) != 0;
    if ( v17 && v18 && (v16 & 0xFFFFFFFFFFFFFFF8LL) == v77 )
      goto LABEL_45;
    if ( !v15 )
      goto LABEL_43;
    goto LABEL_13;
  }
  v74 = 0;
  v18 = 0;
  if ( v15 )
  {
LABEL_13:
    if ( (v15 & 4) != 0 )
    {
LABEL_14:
      v19 = v15 & 0xFFFFFFFFFFFFFFF8LL;
      v81 = v19 != 0;
      v20 = v19;
      v21 = v19 != 0 && v18;
      if ( v16 && (v16 & 4) != 0 )
        goto LABEL_16;
      if ( v21 )
      {
        v64 = v14;
        v67 = v12;
        v70 = v9;
        v43 = (*(__int64 (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v19 + 40LL))(v19, a1);
        v23 = 0;
        v25 = 0;
        v9 = v70;
        v12 = v67;
        v14 = v64;
        if ( !v43 )
          return 0;
        goto LABEL_66;
      }
LABEL_23:
      if ( a2 && v77 && v74 && (v9 <= 0 || !v76) && (v8 <= 0 || !v71) )
      {
        v27 = -1;
        if ( v12 != -1 )
          v27 = v9 - v72 + (v12 & 0x3FFFFFFFFFFFFFFFLL);
        v28 = v80 & (v76 == 0);
        if ( v14 == -1 )
        {
          if ( !v28 )
          {
LABEL_39:
            if ( a3 )
            {
              v31 = a5[5];
              v84 = v14;
              v32 = a5[6];
              v33 = a5[7];
              v34 = a5[8];
              v83 = v74;
              v85 = v31;
              v35 = a4[8];
              v86 = v32;
              v36 = a4[7];
              v87 = v33;
              v37 = a4[6];
              v88 = v34;
              v38 = a4[5];
            }
            else
            {
              v84 = v14;
              v35 = 0;
              v36 = 0;
              v85 = 0;
              v37 = 0;
              v83 = v74;
              v38 = 0;
              v86 = 0;
              v87 = 0;
              v88 = 0;
            }
            v82[2] = v38;
            v82[5] = v35;
            v39 = *a2;
            v82[1] = v12;
            v82[3] = v37;
            v82[4] = v36;
            v82[0] = v77;
            return (unsigned __int8)sub_CF4D50(v39, (__int64)v82, (__int64)&v83, (__int64)(a2 + 1), 0) != 0;
          }
          v29 = -1;
        }
        else
        {
          v29 = (v14 & 0x3FFFFFFFFFFFFFFFLL) + v8 - v72;
          if ( !v28 )
          {
LABEL_34:
            if ( !v71 && v79 )
            {
              v30 = 0xBFFFFFFFFFFFFFFELL;
              if ( v29 <= 0x3FFFFFFFFFFFFFFBLL )
                v30 = v29;
              v14 = v30;
            }
            goto LABEL_39;
          }
        }
        v12 = 0xBFFFFFFFFFFFFFFELL;
        if ( v27 <= 0x3FFFFFFFFFFFFFFBLL )
          v12 = v27;
        goto LABEL_34;
      }
      return 1;
    }
    goto LABEL_42;
  }
LABEL_43:
  if ( (v16 & 4) == 0 )
    goto LABEL_23;
  v81 = 0;
  v20 = 0;
  v21 = 0;
LABEL_16:
  v22 = v16 & 0xFFFFFFFFFFFFFFF8LL;
  v23 = v22;
  if ( v21 )
  {
    v60 = v14;
    v61 = v12;
    v62 = v9;
    v65 = v17;
    v68 = a1;
    v24 = (*(__int64 (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v20 + 40LL))(v20, a1);
    a1 = v68;
    v17 = v65;
    v23 = v22;
    v9 = v62;
    v12 = v61;
    v14 = v60;
    if ( !v24 )
      return 0;
  }
  v25 = v22 != 0;
  if ( v22 && v17 )
  {
    v63 = v14;
    v66 = v12;
    v69 = v9;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v22 + 40LL))(v22, a1) )
    {
      v14 = v63;
      v12 = v66;
      v9 = v69;
      v26 = v22 == v20 && v81;
      goto LABEL_22;
    }
    return 0;
  }
LABEL_66:
  v26 = v25 && v23 == v20 && v81;
LABEL_22:
  if ( !v26 )
    goto LABEL_23;
LABEL_45:
  if ( v73 )
    goto LABEL_23;
  if ( !v80 || !v79 )
    return 1;
  v41 = v8;
  if ( v9 >= v8 )
    v41 = v9;
  v42 = v12 & 0x3FFFFFFFFFFFFFFFLL;
  if ( v9 > v8 )
    v42 = v14 & 0x3FFFFFFFFFFFFFFFLL;
  return v72 + v42 > v41;
}
