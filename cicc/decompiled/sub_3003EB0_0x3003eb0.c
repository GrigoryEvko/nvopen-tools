// Function: sub_3003EB0
// Address: 0x3003eb0
//
__int64 __fastcall sub_3003EB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdi
  __int64 (*v8)(void); // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 (*v12)(void); // rdx
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // r10d
  int v17; // edi
  unsigned int i; // eax
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // r10d
  unsigned int j; // eax
  __int64 v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  int v34; // r11d
  unsigned int k; // eax
  __int64 v36; // r8
  unsigned int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rsi
  __int64 v44; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 **v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  void **v70; // rax
  __int64 **v71; // rsi
  __int64 v72; // [rsp+0h] [rbp-1A0h] BYREF
  void **v73; // [rsp+8h] [rbp-198h]
  unsigned int v74; // [rsp+10h] [rbp-190h]
  unsigned int v75; // [rsp+14h] [rbp-18Ch]
  char v76; // [rsp+1Ch] [rbp-184h]
  _BYTE v77[16]; // [rsp+20h] [rbp-180h] BYREF
  _BYTE v78[8]; // [rsp+30h] [rbp-170h] BYREF
  unsigned __int64 v79; // [rsp+38h] [rbp-168h]
  int v80; // [rsp+44h] [rbp-15Ch]
  int v81; // [rsp+48h] [rbp-158h]
  char v82; // [rsp+4Ch] [rbp-154h]
  _BYTE v83[16]; // [rsp+50h] [rbp-150h] BYREF
  _QWORD *v84[5]; // [rsp+60h] [rbp-140h] BYREF
  __int64 v85; // [rsp+88h] [rbp-118h]
  __int64 v86; // [rsp+90h] [rbp-110h]
  __int64 v87; // [rsp+98h] [rbp-108h]
  int v88; // [rsp+A0h] [rbp-100h]
  __int64 v89; // [rsp+A8h] [rbp-F8h]
  __int64 v90; // [rsp+B0h] [rbp-F0h]
  __int64 v91; // [rsp+B8h] [rbp-E8h]
  __int64 v92; // [rsp+C0h] [rbp-E0h]
  unsigned int v93; // [rsp+C8h] [rbp-D8h]
  __int64 v94; // [rsp+D0h] [rbp-D0h]
  char *v95; // [rsp+D8h] [rbp-C8h]
  __int64 v96; // [rsp+E0h] [rbp-C0h]
  int v97; // [rsp+E8h] [rbp-B8h]
  char v98; // [rsp+ECh] [rbp-B4h]
  char v99; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v100; // [rsp+130h] [rbp-70h]
  __int64 v101; // [rsp+138h] [rbp-68h]
  __int64 v102; // [rsp+140h] [rbp-60h]
  unsigned int v103; // [rsp+148h] [rbp-58h]
  __int64 v104; // [rsp+150h] [rbp-50h]
  __int64 v105; // [rsp+158h] [rbp-48h]
  __int64 v106; // [rsp+160h] [rbp-40h]
  unsigned int v107; // [rsp+168h] [rbp-38h]

  v7 = *(_QWORD *)(a3 + 16);
  v84[0] = (_QWORD *)a3;
  v8 = *(__int64 (**)(void))(*(_QWORD *)v7 + 128LL);
  v9 = 0;
  if ( v8 != sub_2DAC790 )
  {
    v9 = (_QWORD *)v8();
    v7 = *(_QWORD *)(a3 + 16);
  }
  v84[1] = v9;
  v10 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 200LL))(v7);
  v11 = *(_QWORD *)(a3 + 16);
  v84[2] = v10;
  v12 = *(__int64 (**)(void))(*(_QWORD *)v11 + 216LL);
  v13 = 0;
  if ( v12 != sub_2F391C0 )
    v13 = (_QWORD *)v12();
  v14 = *(unsigned int *)(a4 + 88);
  v15 = *(_QWORD *)(a4 + 72);
  v84[3] = v13;
  v84[4] = *(_QWORD **)(a3 + 32);
  if ( (_DWORD)v14 )
  {
    v16 = 1;
    v17 = v14 - 1;
    for ( i = (v14 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_501EB18 >> 9) ^ ((unsigned int)&unk_501EB18 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v17 & v20 )
    {
      v19 = v15 + 24LL * i;
      if ( *(_UNKNOWN **)v19 == &unk_501EB18 && a3 == *(_QWORD *)(v19 + 8) )
        break;
      if ( *(_QWORD *)v19 == -4096 && *(_QWORD *)(v19 + 8) == -4096 )
        goto LABEL_11;
      v20 = v16 + i;
      ++v16;
    }
    v22 = v15 + 24 * v14;
    if ( v22 != v19 )
    {
      v23 = *(_QWORD *)(*(_QWORD *)(v19 + 16) + 24LL);
      if ( v23 )
        v23 += 8;
      v85 = v23;
      goto LABEL_18;
    }
    v85 = 0;
  }
  else
  {
LABEL_11:
    v85 = 0;
    v19 = v15 + 24LL * (unsigned int)v14;
    if ( !(_DWORD)v14 )
      goto LABEL_12;
    v17 = v14 - 1;
  }
  v22 = v19;
LABEL_18:
  v24 = 1;
  for ( j = v17
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_501EAD0 >> 9) ^ ((unsigned int)&unk_501EAD0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v17 & v27 )
  {
    v26 = v15 + 24LL * j;
    if ( *(_UNKNOWN **)v26 == &unk_501EAD0 && a3 == *(_QWORD *)(v26 + 8) )
      break;
    if ( *(_QWORD *)v26 == -4096 && *(_QWORD *)(v26 + 8) == -4096 )
      goto LABEL_12;
    v27 = v24 + j;
    ++v24;
  }
  if ( v26 != v22 )
  {
    v21 = *(_QWORD *)(*(_QWORD *)(v26 + 16) + 24LL);
    if ( v21 )
      v21 += 8;
    goto LABEL_26;
  }
LABEL_12:
  v21 = 0;
LABEL_26:
  v86 = v21;
  v28 = *(_QWORD *)(a3 + 8);
  v87 = 0;
  LODWORD(v28) = *(_DWORD *)(v28 + 648);
  v89 = 0;
  v90 = 0;
  v88 = v28;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = &v99;
  v96 = 8;
  v97 = 0;
  v98 = 1;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v29 = sub_2EB2140(a4, &qword_50209D0, a3);
  v30 = *(_QWORD *)a3;
  v31 = *(_QWORD *)(v29 + 8);
  v32 = *(unsigned int *)(v31 + 88);
  v33 = *(_QWORD *)(v31 + 72);
  if ( !(_DWORD)v32 )
    goto LABEL_56;
  v34 = 1;
  for ( k = (v32 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F86540 >> 9) ^ ((unsigned int)&unk_4F86540 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4)))); ; k = (v32 - 1) & v37 )
  {
    v36 = v33 + 24LL * k;
    if ( *(_UNKNOWN **)v36 == &unk_4F86540 && v30 == *(_QWORD *)(v36 + 8) )
      break;
    if ( *(_QWORD *)v36 == -4096 && *(_QWORD *)(v36 + 8) == -4096 )
      goto LABEL_56;
    v37 = v34 + k;
    ++v34;
  }
  if ( v36 == v33 + 24 * v32 )
  {
LABEL_56:
    v38 = 0;
  }
  else
  {
    v38 = *(_QWORD *)(*(_QWORD *)(v36 + 16) + 24LL);
    if ( v38 )
      v38 += 8;
  }
  v87 = v38;
  if ( (unsigned __int8)sub_B2D610(v30, 48) )
    v88 = 0;
  if ( !(unsigned __int8)sub_3000EC0(v84, 48, v39, v40, v41, v42) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_39;
  }
  sub_2EAFFB0((__int64)&v72);
  sub_2FF9250((__int64)&v72, (__int64)&unk_501EAD0, v46, v47, v48, v49);
  sub_2FF9250((__int64)&v72, (__int64)&unk_501EB18, v50, v51, v52, v53);
  sub_2FF9250((__int64)&v72, (__int64)qword_501FE48, v54, v55, v56, v57);
  sub_2FF9250((__int64)&v72, (__int64)&qword_50208B0, v58, v59, v60, v61);
  sub_2FF9250((__int64)&v72, (__int64)&qword_5025C20, v62, v63, v64, v65);
  if ( v80 == v81 )
  {
    if ( v76 )
    {
      v70 = v73;
      v71 = (__int64 **)&v73[v75];
      v67 = v75;
      v66 = (__int64 **)v73;
      if ( v73 != (void **)v71 )
      {
        while ( *v66 != &qword_4F82400 )
        {
          if ( v71 == ++v66 )
          {
LABEL_47:
            while ( *v70 != &unk_4F82408 )
            {
              if ( v66 == (__int64 **)++v70 )
                goto LABEL_52;
            }
            goto LABEL_48;
          }
        }
        goto LABEL_48;
      }
      goto LABEL_52;
    }
    if ( sub_C8CA60((__int64)&v72, (__int64)&qword_4F82400) )
      goto LABEL_48;
  }
  if ( !v76 )
  {
LABEL_54:
    sub_C8CC70((__int64)&v72, (__int64)&unk_4F82408, (__int64)v66, v67, v68, v69);
    goto LABEL_48;
  }
  v70 = v73;
  v67 = v75;
  v66 = (__int64 **)&v73[v75];
  if ( v66 != (__int64 **)v73 )
    goto LABEL_47;
LABEL_52:
  if ( v74 <= (unsigned int)v67 )
    goto LABEL_54;
  v75 = v67 + 1;
  *v66 = (__int64 *)&unk_4F82408;
  ++v72;
LABEL_48:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v77, (__int64)&v72);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v83, (__int64)v78);
  if ( !v82 )
    _libc_free(v79);
  if ( !v76 )
    _libc_free((unsigned __int64)v73);
LABEL_39:
  v43 = v107;
  v44 = v105;
  *(_QWORD *)(a3 + 344) |= 0x100uLL;
  sub_C7D6A0(v44, 8 * v43, 4);
  sub_C7D6A0(v101, 8LL * v103, 4);
  if ( !v98 )
    _libc_free((unsigned __int64)v95);
  sub_C7D6A0(v91, 16LL * v93, 8);
  return a1;
}
