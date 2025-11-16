// Function: sub_27B1AE0
// Address: 0x27b1ae0
//
void __fastcall sub_27B1AE0(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 *v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // r14
  int v10; // ecx
  unsigned int v11; // edx
  char *v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // edi
  unsigned int v15; // edx
  char *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v19; // rdx
  unsigned int v20; // r11d
  __int64 v21; // rdx
  __int64 *v22; // rbx
  __int64 v23; // rax
  char *v24; // r12
  unsigned int v25; // r14d
  char *v26; // r15
  __int64 v27; // r10
  __int64 v28; // r8
  char *v29; // r9
  __int64 v30; // r13
  __int64 v31; // rcx
  unsigned int v32; // r11d
  unsigned int v33; // edx
  char *v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // edx
  char *v37; // rdi
  char *v38; // rax
  char *i; // rsi
  char *v40; // rax
  unsigned int v41; // r9d
  unsigned int v42; // edx
  char *v43; // rax
  __int64 v44; // r8
  int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rax
  unsigned int v48; // r8d
  __int64 v49; // rdx
  int v50; // eax
  __int64 v51; // rsi
  int v52; // eax
  int v53; // esi
  __int64 v54; // rax
  int v55; // eax
  int v56; // r9d
  int v57; // eax
  bool v58; // al
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdx
  __int64 v63; // rax
  char *v64; // rbx
  __int64 v65; // rcx
  __int64 v66; // rax
  int v67; // edi
  __int64 v68; // rbx
  __int64 v69; // r14
  __int64 v70; // r13
  __int64 v71; // r12
  int v72; // eax
  int v73; // r8d
  int v74; // edi
  char *v75; // [rsp+0h] [rbp-E0h]
  int v76; // [rsp+8h] [rbp-D8h]
  unsigned int v77; // [rsp+Ch] [rbp-D4h]
  int v78; // [rsp+Ch] [rbp-D4h]
  __int64 v80; // [rsp+20h] [rbp-C0h]
  char *v81; // [rsp+28h] [rbp-B8h]
  char *v83; // [rsp+40h] [rbp-A0h]
  __int64 v84; // [rsp+40h] [rbp-A0h]
  unsigned int v85; // [rsp+48h] [rbp-98h]
  __int64 v86; // [rsp+48h] [rbp-98h]
  __int64 v87; // [rsp+50h] [rbp-90h] BYREF
  __int64 v88; // [rsp+58h] [rbp-88h]
  __int64 v89; // [rsp+60h] [rbp-80h]
  __int64 v90; // [rsp+68h] [rbp-78h]
  __int64 v91; // [rsp+70h] [rbp-70h] BYREF
  void *src; // [rsp+78h] [rbp-68h]
  __int64 v93; // [rsp+80h] [rbp-60h]
  __int64 v94; // [rsp+88h] [rbp-58h]
  __int64 v95; // [rsp+90h] [rbp-50h] BYREF
  char *v96; // [rsp+98h] [rbp-48h]
  __int64 v97; // [rsp+A0h] [rbp-40h]
  __int64 v98; // [rsp+A8h] [rbp-38h]

  v81 = a2;
  v4 = a2 - (char *)a1;
  v80 = a3;
  if ( v4 <= 256 )
    return;
  v5 = v4;
  if ( !a3 )
  {
    v83 = v81;
    goto LABEL_62;
  }
  v75 = (char *)(a1 + 2);
  while ( 2 )
  {
    v91 = 0;
    --v80;
    src = 0;
    v93 = 0;
    v94 = 0;
    sub_27B1670((__int64)&v91, a4);
    v95 = 0;
    v96 = 0;
    v6 = &a1[2 * (v5 >> 5)];
    v97 = 0;
    v98 = 0;
    sub_27B1670((__int64)&v95, (__int64)&v91);
    v7 = a1[2];
    v8 = (__int64)v96;
    v9 = *v6;
    if ( !(_DWORD)v98 )
      goto LABEL_49;
    v10 = v98 - 1;
    v11 = (v98 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v12 = &v96[16 * v11];
    v13 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 == v7 )
    {
LABEL_6:
      v14 = *((_DWORD *)v12 + 2);
    }
    else
    {
      v72 = 1;
      while ( v13 != -4096 )
      {
        v74 = v72 + 1;
        v11 = v10 & (v72 + v11);
        v12 = &v96[16 * v11];
        v13 = *(_QWORD *)v12;
        if ( v7 == *(_QWORD *)v12 )
          goto LABEL_6;
        v72 = v74;
      }
      v14 = 0;
    }
    v15 = v10 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v16 = &v96[16 * v15];
    v17 = *(_QWORD *)v16;
    if ( v9 != *(_QWORD *)v16 )
    {
      v57 = 1;
      while ( v17 != -4096 )
      {
        v73 = v57 + 1;
        v15 = v10 & (v57 + v15);
        v16 = &v96[16 * v15];
        v17 = *(_QWORD *)v16;
        if ( v9 == *(_QWORD *)v16 )
          goto LABEL_8;
        v57 = v73;
      }
      goto LABEL_49;
    }
LABEL_8:
    if ( *((_DWORD *)v16 + 2) <= v14 )
    {
LABEL_49:
      v58 = sub_27AD0E0((__int64)&v95, v7, *((_QWORD *)v81 - 2));
      v18 = *a1;
      if ( v58 )
      {
LABEL_50:
        v60 = a1[1];
        v61 = a1[3];
        *a1 = v7;
        a1[2] = v18;
        a1[1] = v61;
        a1[3] = v60;
        goto LABEL_12;
      }
      if ( sub_27AD0E0((__int64)&v95, v9, v59) )
      {
        v22 = a1;
        goto LABEL_11;
      }
LABEL_52:
      *a1 = v9;
      v62 = v6[1];
      *v6 = v18;
      v63 = a1[1];
      a1[1] = v62;
      v6[1] = v63;
      goto LABEL_12;
    }
    v18 = *a1;
    if ( sub_27AD0E0((__int64)&v95, *v6, *((_QWORD *)v81 - 2)) )
      goto LABEL_52;
    v22 = a1;
    if ( !sub_27AD0E0((__int64)&v95, v7, v19) )
      goto LABEL_50;
LABEL_11:
    *v22 = v21;
    *((_QWORD *)v81 - 2) = v18;
    v23 = v22[1];
    v22[1] = *((_QWORD *)v81 - 1);
    *((_QWORD *)v81 - 1) = v23;
LABEL_12:
    sub_C7D6A0(v8, 16LL * v20, 8);
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    sub_27B1670((__int64)&v95, (__int64)&v91);
    v24 = v96;
    v25 = v98;
    v26 = v75;
    v27 = *a1;
    v28 = *((_QWORD *)v81 - 2);
    v29 = v81 - 16;
    while ( 1 )
    {
      v83 = v26;
      v30 = *(_QWORD *)v26;
      v31 = v28;
      if ( !v25 )
      {
        i = v29;
        if ( v26 >= v29 )
          break;
        goto LABEL_28;
      }
      v32 = v25 - 1;
      v33 = (v25 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v34 = &v24[16 * v33];
      v35 = *(_QWORD *)v34;
      if ( v30 == *(_QWORD *)v34 )
      {
LABEL_16:
        v36 = *((_DWORD *)v34 + 2);
      }
      else
      {
        v55 = 1;
        while ( v35 != -4096 )
        {
          v67 = v55 + 1;
          v33 = v32 & (v55 + v33);
          v34 = &v24[16 * v33];
          v35 = *(_QWORD *)v34;
          if ( v30 == *(_QWORD *)v34 )
            goto LABEL_16;
          v55 = v67;
        }
        v36 = 0;
      }
      v85 = v32 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v37 = &v24[16 * v85];
      v38 = v37;
      if ( v27 == *(_QWORD *)v37 )
      {
LABEL_18:
        if ( v36 < *((_DWORD *)v38 + 2) )
          goto LABEL_13;
      }
      else
      {
        v51 = *(_QWORD *)v37;
        v77 = v32 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v52 = 1;
        while ( v51 != -4096 )
        {
          v53 = v52 + 1;
          v54 = v32 & (v77 + v52);
          v76 = v53;
          v77 = v54;
          v38 = &v24[16 * v54];
          v51 = *(_QWORD *)v38;
          if ( *(_QWORD *)v38 == v27 )
            goto LABEL_18;
          v52 = v76;
        }
      }
      for ( i = v29; ; i -= 16 )
      {
        v40 = &v24[16 * v85];
        if ( v27 == *(_QWORD *)v37 )
        {
LABEL_23:
          v41 = *((_DWORD *)v40 + 2);
        }
        else
        {
          v48 = v32 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v49 = *(_QWORD *)v37;
          v50 = 1;
          while ( v49 != -4096 )
          {
            v56 = v50 + 1;
            v48 = v32 & (v50 + v48);
            v40 = &v24[16 * v48];
            v49 = *(_QWORD *)v40;
            if ( v27 == *(_QWORD *)v40 )
              goto LABEL_23;
            v50 = v56;
          }
          v41 = 0;
        }
        v42 = v32 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v43 = &v24[16 * v42];
        v44 = *(_QWORD *)v43;
        if ( *(_QWORD *)v43 != v31 )
          break;
LABEL_20:
        if ( *((_DWORD *)v43 + 2) <= v41 )
          goto LABEL_27;
        v31 = *((_QWORD *)i - 2);
      }
      v45 = 1;
      while ( v44 != -4096 )
      {
        v42 = v32 & (v45 + v42);
        v78 = v45 + 1;
        v43 = &v24[16 * v42];
        v44 = *(_QWORD *)v43;
        if ( v31 == *(_QWORD *)v43 )
          goto LABEL_20;
        v45 = v78;
      }
LABEL_27:
      if ( v26 >= i )
        break;
LABEL_28:
      *(_QWORD *)v26 = v31;
      v46 = *((_QWORD *)i + 1);
      v29 = i - 16;
      *(_QWORD *)i = v30;
      v47 = *((_QWORD *)v26 + 1);
      *((_QWORD *)v26 + 1) = v46;
      v28 = *((_QWORD *)i - 2);
      *((_QWORD *)i + 1) = v47;
      v25 = v98;
      v24 = v96;
      v27 = *a1;
LABEL_13:
      v26 += 16;
    }
    sub_C7D6A0((__int64)v24, 16LL * v25, 8);
    sub_C7D6A0((__int64)src, 16LL * (unsigned int)v94, 8);
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    sub_27B1670((__int64)&v95, a4);
    sub_27B1AE0(v26, v81, v80, &v95);
    sub_C7D6A0((__int64)v96, 16LL * (unsigned int)v98, 8);
    v5 = v26 - (char *)a1;
    if ( v26 - (char *)a1 > 256 )
    {
      if ( v80 )
      {
        v81 = v26;
        continue;
      }
LABEL_62:
      v87 = 0;
      v89 = 0;
      v68 = v5 >> 4;
      v88 = 0;
      v90 = 0;
      v69 = ((v5 >> 4) - 2) >> 1;
      sub_27B1670((__int64)&v87, a4);
      v91 = 0;
      src = 0;
      v93 = 0;
      v94 = 0;
      sub_27B1670((__int64)&v91, (__int64)&v87);
      while ( 1 )
      {
        v95 = 0;
        v70 = a1[2 * v69];
        v71 = a1[2 * v69 + 1];
        v96 = 0;
        v97 = 0;
        v98 = 0;
        sub_C7D6A0(0, 0, 8);
        LODWORD(v98) = v94;
        if ( (_DWORD)v94 )
        {
          v96 = (char *)sub_C7D670(16LL * (unsigned int)v94, 8);
          v97 = v93;
          memcpy(v96, src, 16LL * (unsigned int)v98);
        }
        else
        {
          v96 = 0;
          v97 = 0;
        }
        sub_27B1700((__int64)a1, v69, v68, v70, v71, (__int64)&v95);
        sub_C7D6A0((__int64)v96, 16LL * (unsigned int)v98, 8);
        if ( !v69 )
          break;
        --v69;
      }
      sub_C7D6A0((__int64)src, 16LL * (unsigned int)v94, 8);
      v64 = v83;
      do
      {
        v65 = *((_QWORD *)v64 - 2);
        v64 -= 16;
        v66 = *((_QWORD *)v64 + 1);
        *(_QWORD *)v64 = *a1;
        v84 = v65;
        *((_QWORD *)v64 + 1) = a1[1];
        v86 = v66;
        v95 = 0;
        v96 = 0;
        v97 = 0;
        v98 = 0;
        sub_27B1670((__int64)&v95, (__int64)&v87);
        sub_27B1700((__int64)a1, 0, (v64 - (char *)a1) >> 4, v84, v86, (__int64)&v95);
        sub_C7D6A0((__int64)v96, 16LL * (unsigned int)v98, 8);
      }
      while ( v64 - (char *)a1 > 16 );
      sub_C7D6A0(v88, 16LL * (unsigned int)v90, 8);
    }
    break;
  }
}
