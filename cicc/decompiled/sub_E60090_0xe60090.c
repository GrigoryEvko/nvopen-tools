// Function: sub_E60090
// Address: 0xe60090
//
void __fastcall sub_E60090(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // r14
  _DWORD *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r8
  int v13; // r10d
  char v14; // di
  int v15; // r13d
  unsigned int v16; // r12d
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned int v22; // ecx
  unsigned int v23; // eax
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // rcx
  __int64 v33; // rsi
  unsigned int v34; // r11d
  int *v35; // rax
  int v36; // r12d
  unsigned int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  unsigned int v43; // r12d
  __int64 *v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 *v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  _QWORD *v52; // r14
  __int64 v53; // rax
  __int64 v54; // r14
  __int64 v55; // rax
  unsigned int v56; // eax
  int v57; // eax
  unsigned int v58; // eax
  __int64 v59; // rcx
  __int64 v60; // r9
  __int64 v61; // r8
  int v62; // r10d
  unsigned int v63; // r12d
  __int64 v64; // rax
  __int64 v65; // rdx
  int v66; // [rsp+4h] [rbp-6Ch]
  unsigned int v67; // [rsp+4h] [rbp-6Ch]
  _DWORD *v68; // [rsp+8h] [rbp-68h]
  unsigned __int64 v69; // [rsp+10h] [rbp-60h]
  __int64 v71; // [rsp+20h] [rbp-50h]
  unsigned int v72; // [rsp+28h] [rbp-48h]
  int v73; // [rsp+28h] [rbp-48h]
  unsigned int v74; // [rsp+28h] [rbp-48h]
  int v75; // [rsp+28h] [rbp-48h]
  int v76; // [rsp+28h] [rbp-48h]
  __int64 v77; // [rsp+28h] [rbp-48h]
  unsigned int v78; // [rsp+28h] [rbp-48h]
  __int64 v79; // [rsp+30h] [rbp-40h]
  __int64 v80; // [rsp+30h] [rbp-40h]
  unsigned int v81; // [rsp+30h] [rbp-40h]
  unsigned int v82; // [rsp+30h] [rbp-40h]
  unsigned int v83; // [rsp+30h] [rbp-40h]
  unsigned int v84; // [rsp+30h] [rbp-40h]
  __int64 v85; // [rsp+30h] [rbp-40h]
  unsigned int v86; // [rsp+30h] [rbp-40h]
  unsigned int v87; // [rsp+30h] [rbp-40h]
  int v88; // [rsp+30h] [rbp-40h]
  unsigned int v89; // [rsp+30h] [rbp-40h]
  _QWORD *v90; // [rsp+30h] [rbp-40h]
  int v91; // [rsp+30h] [rbp-40h]
  unsigned int v92; // [rsp+30h] [rbp-40h]

  v4 = sub_E5FF70(a1, *(_DWORD *)(a3 + 32));
  v69 = v5;
  if ( v5 <= v4 )
    return;
  v6 = sub_E60040(a1, v4, v5);
  v8 = v7;
  if ( !v7 )
    return;
  v9 = v6;
  v79 = *(_QWORD *)(a3 + 48);
  v66 = *(_DWORD *)(a3 + 36);
  v72 = *(_DWORD *)(a3 + 40);
  v10 = sub_E5F790(a1, *(_DWORD *)(a3 + 32));
  *(_QWORD *)(a3 + 72) = 0;
  v11 = v79;
  v68 = v10;
  v71 = v9 + 24 * v8;
  if ( v9 != v71 )
  {
    v12 = v72;
    v13 = v66;
    v14 = 0;
    do
    {
      v31 = *(_DWORD *)(v9 + 8);
      if ( *(_DWORD *)(a3 + 32) == v31 )
      {
        v15 = *(_DWORD *)(v9 + 12);
        v16 = *(_DWORD *)(v9 + 16);
        if ( !v14 )
          goto LABEL_25;
      }
      else
      {
        v32 = (unsigned int)v68[12];
        v33 = *((_QWORD *)v68 + 4);
        if ( !(_DWORD)v32 )
          goto LABEL_50;
        v34 = (v32 - 1) & (37 * v31);
        v35 = (int *)(v33 + 16LL * v34);
        v36 = *v35;
        if ( v31 != *v35 )
        {
          v57 = 1;
          while ( v36 != -1 )
          {
            v34 = (v32 - 1) & (v57 + v34);
            v91 = v57 + 1;
            v35 = (int *)(v33 + 16LL * v34);
            v36 = *v35;
            if ( v31 == *v35 )
              goto LABEL_23;
            v57 = v91;
          }
LABEL_50:
          if ( v14 )
          {
            v75 = v13;
            v86 = v12;
            v58 = sub_E5F570(a2, v11, *(_QWORD *)v9);
            v61 = v86;
            v62 = v75;
            v63 = v58;
            v64 = *(_QWORD *)(a3 + 72);
            if ( (unsigned __int64)(v64 + 1) > *(_QWORD *)(a3 + 80) )
            {
              sub_C8D290(a3 + 64, (const void *)(a3 + 88), v64 + 1, 1u, v86, v60);
              v64 = *(_QWORD *)(a3 + 72);
              v62 = v75;
              v61 = v86;
            }
            v65 = *(_QWORD *)(a3 + 64);
            v76 = v62;
            v87 = v61;
            *(_BYTE *)(v65 + v64) = 4;
            ++*(_QWORD *)(a3 + 72);
            sub_E5F5F0(v63, (_QWORD *)(a3 + 64), v65, v59, v61, v60);
            v11 = *(_QWORD *)v9;
            v13 = v76;
            v14 = 0;
            v12 = v87;
          }
          else
          {
            v14 = 0;
          }
          goto LABEL_18;
        }
LABEL_23:
        if ( v35 == (int *)(v33 + 16 * v32) )
          goto LABEL_50;
        v15 = v35[1];
        v16 = v35[2];
        if ( !v14 )
        {
LABEL_25:
          if ( v15 == v13 )
          {
            v21 = *(_QWORD *)v9;
            v22 = v16 - v12;
            if ( (int)(v16 - v12) < 0 )
              goto LABEL_27;
LABEL_12:
            v74 = v22;
            v81 = 2 * v22;
            v23 = sub_E5F570(a2, v11, v21);
            v24 = v81;
            v25 = v74;
            v26 = v23;
            if ( v81 <= 7 && v23 <= 0xF )
              goto LABEL_14;
            if ( !v74 )
            {
              v40 = *(_QWORD *)(a3 + 72);
              v41 = v40 + 1;
              if ( (unsigned __int64)(v40 + 1) <= *(_QWORD *)(a3 + 80) )
                goto LABEL_35;
LABEL_32:
              v84 = v26;
              sub_C8D290(a3 + 64, (const void *)(a3 + 88), v41, 1u, v24, v26);
              v40 = *(_QWORD *)(a3 + 72);
              v26 = v84;
              goto LABEL_35;
            }
LABEL_29:
            v38 = *(_QWORD *)(a3 + 72);
            if ( (unsigned __int64)(v38 + 1) > *(_QWORD *)(a3 + 80) )
            {
              v78 = v24;
              v92 = v26;
              sub_C8D290(a3 + 64, (const void *)(a3 + 88), v38 + 1, 1u, v24, v26);
              v38 = *(_QWORD *)(a3 + 72);
              v24 = v78;
              v26 = v92;
            }
            v39 = *(_QWORD *)(a3 + 64);
            v83 = v26;
            *(_BYTE *)(v39 + v38) = 6;
            ++*(_QWORD *)(a3 + 72);
            sub_E5F5F0(v24, (_QWORD *)(a3 + 64), v39, v25, v24, v26);
            v40 = *(_QWORD *)(a3 + 72);
            v26 = v83;
            v41 = v40 + 1;
            if ( (unsigned __int64)(v40 + 1) > *(_QWORD *)(a3 + 80) )
              goto LABEL_32;
LABEL_35:
            v42 = *(_QWORD *)(a3 + 64);
            *(_BYTE *)(v42 + v40) = 3;
            ++*(_QWORD *)(a3 + 72);
            sub_E5F5F0(v26, (_QWORD *)(a3 + 64), v42, v25, v24, v26);
          }
          else
          {
LABEL_8:
            v17 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 32LL * (unsigned int)(v15 - 1) + 24);
            *(_BYTE *)(v17 + 8) |= 8u;
            v18 = *(_QWORD *)(a3 + 72);
            v19 = *(_QWORD *)(*(_QWORD *)(v17 + 24) + 16LL);
            if ( (unsigned __int64)(v18 + 1) > *(_QWORD *)(a3 + 80) )
            {
              v67 = v12;
              v77 = v11;
              v88 = v19;
              sub_C8D290(a3 + 64, (const void *)(a3 + 88), v18 + 1, 1u, v12, v11);
              v18 = *(_QWORD *)(a3 + 72);
              v12 = v67;
              v11 = v77;
              LODWORD(v19) = v88;
            }
            v20 = *(_QWORD *)(a3 + 64);
            v73 = v12;
            v80 = v11;
            *(_BYTE *)(v20 + v18) = 5;
            ++*(_QWORD *)(a3 + 72);
            sub_E5F5F0(v19, (_QWORD *)(a3 + 64), v18, v20, v12, v11);
            LODWORD(v12) = v73;
            v11 = v80;
LABEL_11:
            v21 = *(_QWORD *)v9;
            v22 = v16 - v12;
            if ( (int)(v16 - v12) >= 0 )
              goto LABEL_12;
LABEL_27:
            v82 = 2 * (v12 - v16) + 1;
            v37 = sub_E5F570(a2, v11, v21);
            v24 = v82;
            v26 = v37;
            if ( v37 > 0xF || v82 > 7 )
              goto LABEL_29;
LABEL_14:
            v27 = *(_QWORD *)(a3 + 72);
            v28 = (unsigned int)(16 * v24);
            v29 = (unsigned int)v28 | (unsigned int)v26;
            if ( (unsigned __int64)(v27 + 1) > *(_QWORD *)(a3 + 80) )
            {
              v89 = v29;
              sub_C8D290(a3 + 64, (const void *)(a3 + 88), v27 + 1, 1u, v28, v29);
              v27 = *(_QWORD *)(a3 + 72);
              v29 = v89;
            }
            v30 = *(_QWORD *)(a3 + 64);
            *(_BYTE *)(v30 + v27) = 11;
            ++*(_QWORD *)(a3 + 72);
            sub_E5F5F0(v29, (_QWORD *)(a3 + 64), v30, v25, v28, v29);
          }
          v11 = *(_QWORD *)v9;
          v12 = v16;
          v13 = v15;
          v14 = 1;
          goto LABEL_18;
        }
      }
      if ( v15 != v13 )
        goto LABEL_8;
      if ( v16 != (_DWORD)v12 )
        goto LABEL_11;
LABEL_18:
      v9 += 24;
    }
    while ( v71 != v9 && *(_QWORD *)(a3 + 72) <= 0xFEEBu );
  }
  v85 = v11;
  v43 = sub_E5F570(a2, v11, *(_QWORD *)(a3 + 56));
  v44 = (__int64 *)sub_E60040(a1, v69, v69 + 1);
  v47 = v85;
  v48 = v44;
  if ( v49 )
  {
    v52 = (_QWORD *)*v44;
    v53 = *(_QWORD *)*v44;
    if ( !v53 )
    {
      if ( (*((_BYTE *)v52 + 9) & 0x70) != 0x20 || *((char *)v52 + 8) < 0 )
        goto LABEL_63;
      *((_BYTE *)v52 + 8) |= 8u;
      v53 = sub_E807D0(v52[3]);
      v47 = v85;
      *v52 = v53;
    }
    v54 = *(_QWORD *)(v53 + 8);
    v55 = *(_QWORD *)v47;
    if ( *(_QWORD *)v47 )
    {
LABEL_44:
      if ( *(_QWORD *)(v55 + 8) == v54 )
      {
        v56 = sub_E5F570(a2, v47, *v48);
        if ( v43 > v56 )
          v43 = v56;
      }
      goto LABEL_39;
    }
    if ( (*(_BYTE *)(v47 + 9) & 0x70) == 0x20 && *(char *)(v47 + 8) >= 0 )
    {
      *(_BYTE *)(v47 + 8) |= 8u;
      v90 = (_QWORD *)v47;
      v55 = sub_E807D0(*(_QWORD *)(v47 + 24));
      v47 = (__int64)v90;
      *v90 = v55;
      goto LABEL_44;
    }
LABEL_63:
    BUG();
  }
LABEL_39:
  v50 = *(_QWORD *)(a3 + 72);
  if ( (unsigned __int64)(v50 + 1) > *(_QWORD *)(a3 + 80) )
  {
    sub_C8D290(a3 + 64, (const void *)(a3 + 88), v50 + 1, 1u, v46, v47);
    v50 = *(_QWORD *)(a3 + 72);
  }
  v51 = *(_QWORD *)(a3 + 64);
  *(_BYTE *)(v51 + v50) = 4;
  ++*(_QWORD *)(a3 + 72);
  sub_E5F5F0(v43, (_QWORD *)(a3 + 64), v51, v45, v46, v47);
}
