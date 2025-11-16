// Function: sub_3968AE0
// Address: 0x3968ae0
//
void __fastcall sub_3968AE0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 **v5; // rax
  int *v6; // r8
  int v7; // r9d
  __int64 *v8; // rcx
  __int64 *v9; // r13
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rax
  bool v14; // zf
  int *v15; // r12
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rcx
  int v19; // r8d
  __int64 v20; // r9
  __int64 v21; // rcx
  int v22; // r8d
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rdx
  int v31; // r8d
  unsigned int v32; // r9d
  __int64 *v33; // rsi
  __int64 v34; // r10
  __int64 *v35; // rdx
  __int64 v36; // rax
  unsigned int v37; // r9d
  __int64 *v38; // rsi
  __int64 v39; // r10
  __int64 v40; // rdx
  __int64 v41; // rcx
  int v42; // esi
  unsigned __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 *v46; // rcx
  unsigned __int8 v47; // al
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rcx
  int v51; // r8d
  __int64 v52; // r9
  __int64 v53; // r12
  __int64 v54; // rcx
  int v55; // r8d
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // r12
  __int64 v59; // r13
  __int64 v60; // rax
  int v61; // esi
  int v62; // esi
  int v63; // r11d
  int v64; // r11d
  __int64 v65; // r12
  __int64 v66; // rax
  __int64 v67; // rcx
  int v68; // r8d
  __int64 v69; // r9
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // r12
  __int64 v73; // rcx
  int v74; // r8d
  __int64 v75; // r9
  __int64 **v76; // [rsp+28h] [rbp-158h]
  int *v77; // [rsp+30h] [rbp-150h]
  int v78; // [rsp+30h] [rbp-150h]
  __int64 **v79; // [rsp+38h] [rbp-148h]
  char v80; // [rsp+4Eh] [rbp-132h] BYREF
  char v81; // [rsp+4Fh] [rbp-131h] BYREF
  __int64 *v82; // [rsp+50h] [rbp-130h] BYREF
  __int64 v83; // [rsp+58h] [rbp-128h] BYREF
  _QWORD v84[4]; // [rsp+60h] [rbp-120h] BYREF
  __int64 v85; // [rsp+80h] [rbp-100h] BYREF
  unsigned __int64 v86; // [rsp+88h] [rbp-F8h]
  __int64 v87; // [rsp+90h] [rbp-F0h]
  int v88; // [rsp+98h] [rbp-E8h]
  __int64 v89; // [rsp+A0h] [rbp-E0h] BYREF
  int v90; // [rsp+A8h] [rbp-D8h] BYREF
  unsigned __int64 v91; // [rsp+B0h] [rbp-D0h]
  int *v92; // [rsp+B8h] [rbp-C8h]
  int *v93; // [rsp+C0h] [rbp-C0h]
  __int64 v94; // [rsp+C8h] [rbp-B8h]
  int *v95[2]; // [rsp+D0h] [rbp-B0h] BYREF
  int v96; // [rsp+E0h] [rbp-A0h]
  _BYTE v97[8]; // [rsp+E8h] [rbp-98h] BYREF
  __int64 v98; // [rsp+F0h] [rbp-90h]
  unsigned __int64 v99; // [rsp+F8h] [rbp-88h]

  v4 = *(unsigned int *)(a2 + 56);
  v92 = &v90;
  v93 = &v90;
  v84[1] = &v85;
  v82 = &v85;
  v5 = *(__int64 ***)(a2 + 48);
  v84[0] = &v89;
  v90 = 0;
  v91 = 0;
  v94 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v84[2] = a1;
  v76 = &v5[v4];
  if ( v5 == v76 )
  {
    *(_DWORD *)(a2 + 56) = 0;
    goto LABEL_70;
  }
  v79 = v5;
  while ( 2 )
  {
    v8 = *(__int64 **)a2;
    v9 = *v79;
    v10 = *(_BYTE *)(*(_QWORD *)a2 + 16LL);
    v11 = **v79;
    if ( v10 > 0x17u )
    {
      if ( !(unsigned __int8)sub_3968710((__int64)a1, v8[5], v11, v8) )
        goto LABEL_4;
    }
    else
    {
      if ( v10 != 17 )
        BUG();
      v12 = *(_QWORD *)(v8[3] + 80);
      if ( v12 )
        v12 -= 24;
      if ( !(unsigned __int8)sub_3968710((__int64)a1, v12, v11, v8) )
      {
LABEL_4:
        *((_BYTE *)v9 + 192) = 1;
        goto LABEL_5;
      }
    }
    v80 = 0;
    v13 = sub_39677F0(v84, *v9, &v80);
    if ( v13 )
    {
      v14 = v80 == 0;
      *((_BYTE *)v9 + 192) = 1;
      if ( !v14 )
        *(_BYTE *)(v13 + 193) = 0;
      goto LABEL_5;
    }
    v77 = v92;
    v15 = &v90;
    if ( v92 == &v90 )
    {
LABEL_37:
      if ( !*((_BYTE *)v9 + 192) )
        goto LABEL_38;
      goto LABEL_5;
    }
    while ( 1 )
    {
      v16 = sub_220EF80((__int64)v15);
      v17 = *(_QWORD *)(v16 + 40);
      if ( *(_BYTE *)(v17 + 192) )
      {
        v15 = (int *)v16;
        goto LABEL_36;
      }
      if ( sub_15CC8F0(a1[3], *(_QWORD *)v17, *v9)
        && !(unsigned __int8)sub_3968710((__int64)a1, *(_QWORD *)v17, *v9, *(__int64 **)a2) )
      {
        break;
      }
      if ( sub_15CC8F0(a1[3], *v9, *(_QWORD *)v17)
        && !(unsigned __int8)sub_3968710((__int64)a1, *v9, *(_QWORD *)v17, *(__int64 **)a2) )
      {
        sub_3967AE0((__int64 *)&v82, (__int64)v9, v17, v21, v22, v23);
        goto LABEL_38;
      }
      v24 = *v9;
      v25 = *(_QWORD *)v17;
      v26 = *(_QWORD *)(*(_QWORD *)(*v9 + 56) + 80LL);
      if ( v26 )
      {
        v27 = v26 - 24;
        if ( v24 == v27 || v25 == v27 )
        {
          v83 = v27;
LABEL_49:
          v46 = *(__int64 **)a2;
          v47 = *(_BYTE *)(*(_QWORD *)a2 + 16LL);
          if ( v47 <= 0x17u )
          {
            if ( v47 != 17 )
              BUG();
            v57 = *(_QWORD *)(v46[3] + 80);
            if ( !v57 )
            {
LABEL_52:
              if ( !(unsigned __int8)sub_3968710((__int64)a1, v27, *(_QWORD *)v17, v46)
                && !(unsigned __int8)sub_3968710((__int64)a1, v83, *v9, *(__int64 **)a2) )
              {
                v81 = 0;
                v49 = sub_39677F0(v84, v83, &v81);
                v53 = v49;
                if ( v49 )
                {
                  sub_3967AE0((__int64 *)&v82, v49, (__int64)v9, v50, v51, v52);
                  sub_3967AE0((__int64 *)&v82, v53, v17, v54, v55, v56);
                  goto LABEL_37;
                }
                v78 = *((_DWORD *)sub_3967280((__int64)(a1 + 33), &v83) + 2);
                sub_3963E30((__int64)v95, (__int64)a1, (__int64 *)a2, v83, 1);
                v65 = sub_22077B0(0xD8u);
                v66 = v83;
                *(_DWORD *)(v65 + 24) = v78;
                *(_QWORD *)(v65 + 16) = v66;
                *(int **)(v65 + 32) = v95[0];
                *(int **)(v65 + 40) = v95[1];
                *(_DWORD *)(v65 + 48) = v96;
                sub_16CCEE0((_QWORD *)(v65 + 56), v65 + 96, 8, (__int64)v97);
                *(_QWORD *)(v65 + 160) = v65 + 176;
                *(_QWORD *)(v65 + 168) = 0x400000000LL;
                *(_WORD *)(v65 + 208) = 0;
                sub_2208C80((_QWORD *)v65, (__int64)(a1 + 17));
                ++a1[19];
                if ( v99 != v98 )
                  _libc_free(v99);
                v70 = a1[18];
                v71 = (__int64)v9;
                *(_BYTE *)(v70 + 209) = 1;
                v72 = v70 + 16;
                v9 = (__int64 *)(v70 + 16);
                sub_3967AE0((__int64 *)&v82, v70 + 16, v71, v67, v68, v69);
                sub_3967AE0((__int64 *)&v82, v72, v17, v73, v74, v75);
                goto LABEL_38;
              }
              goto LABEL_35;
            }
            v48 = v57 - 24;
          }
          else
          {
            v48 = v46[5];
          }
          if ( v48 != v27 )
            goto LABEL_52;
          goto LABEL_35;
        }
      }
      else if ( !v25 )
      {
        goto LABEL_35;
      }
      v28 = a1[3];
      v29 = *(_QWORD *)(v28 + 32);
      v30 = *(unsigned int *)(v28 + 48);
      if ( !(_DWORD)v30 )
        goto LABEL_35;
      v31 = v30 - 1;
      v32 = (v30 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v33 = (__int64 *)(v29 + 16LL * v32);
      v34 = *v33;
      if ( v24 == *v33 )
      {
LABEL_25:
        v35 = (__int64 *)(v29 + 16 * v30);
        if ( v35 != v33 )
        {
          v36 = v33[1];
          goto LABEL_27;
        }
      }
      else
      {
        v61 = 1;
        while ( v34 != -8 )
        {
          v64 = v61 + 1;
          v32 = v31 & (v61 + v32);
          v33 = (__int64 *)(v29 + 16LL * v32);
          v34 = *v33;
          if ( v24 == *v33 )
            goto LABEL_25;
          v61 = v64;
        }
        v35 = (__int64 *)(v29 + 16 * v30);
      }
      v36 = 0;
LABEL_27:
      v37 = v31 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v38 = (__int64 *)(v29 + 16LL * v37);
      v39 = *v38;
      if ( v25 != *v38 )
      {
        v62 = 1;
        while ( v39 != -8 )
        {
          v63 = v62 + 1;
          v37 = v31 & (v62 + v37);
          v38 = (__int64 *)(v29 + 16LL * v37);
          v39 = *v38;
          if ( v25 == *v38 )
            goto LABEL_28;
          v62 = v63;
        }
        goto LABEL_35;
      }
LABEL_28:
      if ( v38 != v35 )
      {
        v40 = v38[1];
        if ( v36 )
        {
          if ( v40 )
          {
            while ( v36 != v40 )
            {
              if ( *(_DWORD *)(v36 + 16) < *(_DWORD *)(v40 + 16) )
              {
                v41 = v36;
                v36 = v40;
                v40 = v41;
              }
              v36 = *(_QWORD *)(v36 + 8);
              if ( !v36 )
                goto LABEL_35;
            }
            v27 = *(_QWORD *)v36;
            v83 = v27;
            if ( v27 )
              goto LABEL_49;
          }
        }
      }
LABEL_35:
      v15 = (int *)sub_220EF80((__int64)v15);
LABEL_36:
      if ( v77 == v15 )
        goto LABEL_37;
    }
    sub_3967AE0((__int64 *)&v82, v17, (__int64)v9, v18, v19, v20);
    if ( *((_BYTE *)v9 + 192) )
      goto LABEL_5;
LABEL_38:
    v42 = *((_DWORD *)v9 + 2);
    v43 = v91;
    v6 = &v90;
    LODWORD(v83) = v42;
    if ( !v91 )
      goto LABEL_45;
    do
    {
      while ( 1 )
      {
        v44 = *(_QWORD *)(v43 + 16);
        v45 = *(_QWORD *)(v43 + 24);
        if ( v42 <= *(_DWORD *)(v43 + 32) )
          break;
        v43 = *(_QWORD *)(v43 + 24);
        if ( !v45 )
          goto LABEL_43;
      }
      v6 = (int *)v43;
      v43 = *(_QWORD *)(v43 + 16);
    }
    while ( v44 );
LABEL_43:
    if ( v6 == &v90 || v42 < v6[8] )
    {
LABEL_45:
      v95[0] = (int *)&v83;
      v6 = (int *)sub_39636C0(&v89, (__int64)v6, v95);
    }
    *((_QWORD *)v6 + 5) = v9;
LABEL_5:
    if ( v76 != ++v79 )
      continue;
    break;
  }
  v58 = (__int64)v92;
  for ( *(_DWORD *)(a2 + 56) = 0; (int *)v58 != &v90; v58 = sub_220EEE0(v58) )
  {
    while ( 1 )
    {
      v59 = *(_QWORD *)(v58 + 40);
      if ( !*(_BYTE *)(v59 + 192) )
        break;
      v58 = sub_220EEE0(v58);
      if ( (int *)v58 == &v90 )
        goto LABEL_70;
    }
    v60 = *(unsigned int *)(a2 + 56);
    if ( (unsigned int)v60 >= *(_DWORD *)(a2 + 60) )
    {
      sub_16CD150(a2 + 48, (const void *)(a2 + 64), 0, 8, (int)v6, v7);
      v60 = *(unsigned int *)(a2 + 56);
    }
    *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v60) = v59;
    ++*(_DWORD *)(a2 + 56);
  }
LABEL_70:
  j___libc_free_0(v86);
  sub_3961540(v91);
}
