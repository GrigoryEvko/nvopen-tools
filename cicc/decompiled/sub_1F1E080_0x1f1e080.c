// Function: sub_1F1E080
// Address: 0x1f1e080
//
void __fastcall sub_1F1E080(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 *v10; // r12
  __int64 *v11; // rdi
  unsigned int v12; // r9d
  unsigned int v13; // eax
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r14
  int v21; // ecx
  unsigned __int8 v22; // al
  __int64 v23; // rsi
  unsigned __int8 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // r8d
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rcx
  int *v33; // rbx
  __int64 *v34; // r12
  int v35; // r10d
  __int64 v36; // r15
  unsigned __int64 v37; // rdx
  unsigned int v38; // eax
  __int64 v39; // r14
  __int64 v40; // r11
  unsigned int v41; // eax
  __int64 v42; // rcx
  __int64 v43; // r12
  int v44; // r15d
  __int64 v45; // rax
  int i; // r12d
  int v47; // r8d
  __int64 v48; // rdi
  __int64 v49; // rdx
  int v50; // eax
  _QWORD *v51; // rdi
  int v52; // r13d
  _QWORD *v53; // r15
  __int64 v54; // r9
  int v55; // ebx
  unsigned __int64 v56; // rcx
  unsigned int v57; // eax
  __int64 v58; // r12
  __int64 v59; // rsi
  int v60; // r9d
  __int64 v61; // rcx
  __int64 *v62; // rax
  int v63; // esi
  __int64 *v64; // rdi
  __int64 v65; // rdx
  unsigned int v66; // r12d
  unsigned __int64 v67; // rax
  unsigned int v68; // eax
  __int64 v69; // rsi
  _QWORD *v70; // rdi
  __int64 v71; // rdx
  _DWORD *j; // rax
  __int64 v73; // rdx
  _QWORD *v74; // rdi
  _QWORD *v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // rdi
  _QWORD *v78; // rsi
  _QWORD *v79; // rdx
  __int64 v80; // rsi
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  unsigned __int64 v84; // r9
  __int64 v85; // [rsp+8h] [rbp-F8h]
  __int64 v86; // [rsp+18h] [rbp-E8h]
  unsigned int v87; // [rsp+20h] [rbp-E0h]
  int v88; // [rsp+24h] [rbp-DCh]
  unsigned int v89; // [rsp+24h] [rbp-DCh]
  __int64 v90; // [rsp+28h] [rbp-D8h]
  _QWORD *v91; // [rsp+28h] [rbp-D8h]
  __int64 v92; // [rsp+28h] [rbp-D8h]
  int v93; // [rsp+28h] [rbp-D8h]
  __int64 v94; // [rsp+28h] [rbp-D8h]
  __int64 v95; // [rsp+30h] [rbp-D0h]
  __int64 v96; // [rsp+30h] [rbp-D0h]
  __int64 v97; // [rsp+30h] [rbp-D0h]
  __int64 v98; // [rsp+38h] [rbp-C8h]
  __int64 v99; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v100[2]; // [rsp+48h] [rbp-B8h] BYREF
  _BYTE v101[32]; // [rsp+58h] [rbp-A8h] BYREF
  int v102; // [rsp+78h] [rbp-88h]
  __int64 *v103; // [rsp+80h] [rbp-80h] BYREF
  __int64 v104; // [rsp+88h] [rbp-78h]
  _BYTE v105[112]; // [rsp+90h] [rbp-70h] BYREF

  v6 = a1;
  v7 = a2;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v9 = *(__int64 **)(v8 + 64);
  v10 = &v9[*(unsigned int *)(v8 + 72)];
  v95 = a1 + 208;
  if ( v9 != v10 )
  {
    v90 = a2;
    while ( 1 )
    {
      v20 = *v9;
      a2 = *(_QWORD *)(*v9 + 8);
      a3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (a2 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        break;
LABEL_15:
      if ( v10 == ++v9 )
      {
        v7 = v90;
        goto LABEL_33;
      }
    }
    v21 = *(_DWORD *)(v6 + 388);
    if ( !v21 )
      goto LABEL_18;
    v11 = (__int64 *)(v6 + 200);
    v12 = *(_DWORD *)(v6 + 384);
    v13 = *(_DWORD *)((*(_QWORD *)(v6 + 200) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v6 + 200) >> 1) & 3;
    v14 = (a2 >> 1) & 3 | *(_DWORD *)(a3 + 24);
    if ( v12 )
    {
      if ( v14 < v13 )
        goto LABEL_18;
      v15 = (__int64 *)(v95 + 8LL * (unsigned int)(v21 - 1) + 88);
    }
    else
    {
      if ( v14 < v13 )
        goto LABEL_18;
      v15 = &v11[2 * (unsigned int)(v21 - 1) + 1];
    }
    if ( (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) > v14 )
    {
      if ( v12 )
      {
        v12 = sub_1F15FF0((__int64)v11, a2, 0);
      }
      else
      {
        if ( v14 < (*(_DWORD *)((*(_QWORD *)(v6 + 208) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                  | (unsigned int)(*(__int64 *)(v6 + 208) >> 1) & 3) )
        {
          v80 = 0;
        }
        else
        {
          LODWORD(v80) = 0;
          do
            v80 = (unsigned int)(v80 + 1);
          while ( (*(_DWORD *)((v11[2 * v80 + 1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                 | (unsigned int)(v11[2 * v80 + 1] >> 1) & 3) <= v14 );
          v11 += 2 * v80;
        }
        if ( (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v11 >> 1) & 3) <= v14 )
          v12 = *(_DWORD *)(v6 + 4 * v80 + 344);
      }
LABEL_9:
      a2 = v12;
      sub_1F1A750(v6, v12, (int *)v20, *(_QWORD *)(v20 + 8), 1);
      v16 = *(_QWORD *)(v6 + 72);
      a4 = *(_QWORD *)(v16 + 168);
      v17 = *(_QWORD **)(v16 + 160);
      if ( (_QWORD *)a4 == v17 )
      {
        v18 = &v17[*(unsigned int *)(v16 + 180)];
        if ( v17 == v18 )
        {
          a3 = *(_QWORD *)(v16 + 160);
        }
        else
        {
          do
          {
            if ( v20 == *v17 )
              break;
            ++v17;
          }
          while ( v18 != v17 );
          a3 = (unsigned __int64)v18;
        }
      }
      else
      {
        a2 = v20;
        v98 = *(_QWORD *)(v6 + 72);
        v18 = (_QWORD *)(a4 + 8LL * *(unsigned int *)(v16 + 176));
        v17 = sub_16CC9F0(v16 + 152, v20);
        if ( v20 == *v17 )
        {
          a2 = *(_QWORD *)(v98 + 168);
          if ( a2 == *(_QWORD *)(v98 + 160) )
            a3 = a2 + 8LL * *(unsigned int *)(v98 + 180);
          else
            a3 = a2 + 8LL * *(unsigned int *)(v98 + 176);
        }
        else
        {
          v19 = *(_QWORD *)(v98 + 168);
          if ( v19 != *(_QWORD *)(v98 + 160) )
          {
            a3 = *(unsigned int *)(v98 + 176);
            v17 = (_QWORD *)(v19 + 8 * a3);
            goto LABEL_13;
          }
          v17 = (_QWORD *)(v19 + 8LL * *(unsigned int *)(v98 + 180));
          a3 = (unsigned __int64)v17;
        }
      }
      while ( (_QWORD *)a3 != v17 && *v17 >= 0xFFFFFFFFFFFFFFFELL )
        ++v17;
LABEL_13:
      if ( v17 != v18 )
      {
        a2 = v20;
        sub_1F1BBC0(v6, v20);
      }
      goto LABEL_15;
    }
LABEL_18:
    v12 = 0;
    goto LABEL_9;
  }
LABEL_33:
  if ( (unsigned int)(*(_DWORD *)(v6 + 84) - 1) <= 1 )
    sub_1F1D260(v6, a2, a3, a4, a5, a6);
  v22 = sub_1F180C0(v6, a2, a3, a4, a5, a6);
  v23 = v22;
  v24 = v22;
  sub_1F16B80(v6, v22, v25, v26, v27);
  if ( v24 )
  {
    sub_1F16160(v6);
    sub_1F15810(v6, v23, v81, v82, v83, v84);
  }
  v30 = *(_QWORD *)(v6 + 72);
  v31 = *(_QWORD *)(v30 + 16);
  v32 = *(unsigned int *)(v30 + 64);
  v99 = *(_QWORD *)v31 + 4LL * *(unsigned int *)(v31 + 8);
  if ( v99 != *(_QWORD *)v31 + 4 * v32 )
  {
    v96 = v7;
    v33 = (int *)(*(_QWORD *)v31 + 4 * v32);
    while ( 1 )
    {
      v35 = *v33;
      v36 = *(_QWORD *)(v6 + 16);
      v37 = *(unsigned int *)(v36 + 408);
      v38 = *v33 & 0x7FFFFFFF;
      v39 = v38;
      v40 = 8LL * v38;
      if ( v38 < (unsigned int)v37 )
      {
        v34 = *(__int64 **)(*(_QWORD *)(v36 + 400) + 8LL * v38);
        if ( v34 )
          goto LABEL_40;
      }
      v41 = v38 + 1;
      if ( (unsigned int)v37 < v41 )
      {
        v43 = v41;
        if ( v41 >= v37 )
        {
          if ( v41 > v37 )
          {
            if ( v41 > (unsigned __int64)*(unsigned int *)(v36 + 412) )
            {
              v86 = v40;
              v89 = v41;
              v93 = *v33;
              sub_16CD150(v36 + 400, (const void *)(v36 + 416), v41, 8, v28, v29);
              v37 = *(unsigned int *)(v36 + 408);
              v40 = v86;
              v41 = v89;
              v35 = v93;
            }
            v42 = *(_QWORD *)(v36 + 400);
            v77 = *(_QWORD *)(v36 + 416);
            v78 = (_QWORD *)(v42 + 8 * v43);
            v79 = (_QWORD *)(v42 + 8 * v37);
            if ( v78 != v79 )
            {
              do
                *v79++ = v77;
              while ( v78 != v79 );
              v42 = *(_QWORD *)(v36 + 400);
            }
            *(_DWORD *)(v36 + 408) = v41;
            goto LABEL_44;
          }
        }
        else
        {
          *(_DWORD *)(v36 + 408) = v41;
        }
      }
      v42 = *(_QWORD *)(v36 + 400);
LABEL_44:
      *(_QWORD *)(v42 + v40) = sub_1DBA290(v35);
      v34 = *(__int64 **)(*(_QWORD *)(v36 + 400) + 8 * v39);
      sub_1DBB110((_QWORD *)v36, (__int64)v34);
LABEL_40:
      ++v33;
      sub_1DB4C70((__int64)v34);
      sub_1DB4280(v34);
      if ( (int *)v99 == v33 )
      {
        v7 = v96;
        break;
      }
    }
  }
  if ( v7 )
  {
    *(_DWORD *)(v7 + 8) = 0;
    v44 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v6 + 72) + 16LL) + 8LL) - *(_DWORD *)(*(_QWORD *)(v6 + 72) + 64LL);
    if ( v44 )
    {
      v45 = 0;
      for ( i = 0; i != v44; ++i )
      {
        if ( *(_DWORD *)(v7 + 12) <= (unsigned int)v45 )
        {
          sub_16CD150(v7, (const void *)(v7 + 16), 0, 4, v28, v29);
          v45 = *(unsigned int *)(v7 + 8);
        }
        *(_DWORD *)(*(_QWORD *)v7 + 4 * v45) = i;
        v45 = (unsigned int)(*(_DWORD *)(v7 + 8) + 1);
        *(_DWORD *)(v7 + 8) = v45;
      }
    }
  }
  v102 = 0;
  v100[0] = (unsigned __int64)v101;
  v100[1] = 0x800000000LL;
  sub_3945AE0(v100, 0);
  v48 = *(_QWORD *)(v6 + 72);
  v49 = *(_QWORD *)(v48 + 16);
  v50 = *(_DWORD *)(v48 + 64);
  v88 = *(_DWORD *)(v49 + 8) - v50;
  if ( v88 )
  {
    v97 = v7;
    v51 = (_QWORD *)v6;
    v52 = 0;
    v53 = v51;
    while ( 1 )
    {
      v54 = v53[2];
      v55 = *(_DWORD *)(*(_QWORD *)v49 + 4LL * (unsigned int)(v52 + v50));
      v56 = *(unsigned int *)(v54 + 408);
      v57 = v55 & 0x7FFFFFFF;
      v58 = v55 & 0x7FFFFFFF;
      if ( (v55 & 0x7FFFFFFFu) >= (unsigned int)v56 )
        break;
      v59 = *(_QWORD *)(*(_QWORD *)(v54 + 400) + 8LL * v57);
      if ( !v59 )
        break;
LABEL_58:
      v103 = (__int64 *)v105;
      v104 = 0x800000000LL;
      sub_1DBEB50(v54, v59, (__int64)&v103);
      v61 = *(_QWORD *)(v53[3] + 312LL);
      v62 = v103;
      v63 = *(_DWORD *)(v61 + 4 * v58);
      v64 = &v103[(unsigned int)v104];
      if ( !v63 )
        v63 = v55;
      if ( v103 != v64 )
      {
        while ( 1 )
        {
          v65 = *v62++;
          *(_DWORD *)(v61 + 4LL * (*(_DWORD *)(v65 + 112) & 0x7FFFFFFF)) = v63;
          if ( v62 == v64 )
            break;
          v61 = *(_QWORD *)(v53[3] + 312LL);
        }
      }
      if ( v97 )
      {
        v66 = *(_DWORD *)(*(_QWORD *)(v53[9] + 16LL) + 8LL) - *(_DWORD *)(v53[9] + 64LL);
        v67 = *(unsigned int *)(v97 + 8);
        if ( v66 >= v67 )
        {
          if ( v66 > v67 )
          {
            if ( v66 > (unsigned __int64)*(unsigned int *)(v97 + 12) )
            {
              sub_16CD150(v97, (const void *)(v97 + 16), v66, 4, v47, v60);
              v67 = *(unsigned int *)(v97 + 8);
            }
            v71 = *(_QWORD *)v97 + 4LL * v66;
            for ( j = (_DWORD *)(*(_QWORD *)v97 + 4 * v67); (_DWORD *)v71 != j; ++j )
              *j = v52;
            *(_DWORD *)(v97 + 8) = v66;
          }
        }
        else
        {
          *(_DWORD *)(v97 + 8) = v66;
        }
      }
      if ( v103 != (__int64 *)v105 )
        _libc_free((unsigned __int64)v103);
      v48 = v53[9];
      if ( ++v52 == v88 )
      {
        v6 = (__int64)v53;
        goto LABEL_83;
      }
      v50 = *(_DWORD *)(v48 + 64);
      v49 = *(_QWORD *)(v48 + 16);
    }
    v68 = v57 + 1;
    if ( (unsigned int)v56 < v68 )
    {
      v73 = v68;
      if ( v68 >= v56 )
      {
        if ( v68 > v56 )
        {
          if ( v68 > (unsigned __int64)*(unsigned int *)(v54 + 412) )
          {
            v87 = v68;
            v85 = v53[2];
            v94 = v68;
            sub_16CD150(v54 + 400, (const void *)(v54 + 416), v68, 8, v47, v54);
            v54 = v85;
            v68 = v87;
            v73 = v94;
            v56 = *(unsigned int *)(v85 + 408);
          }
          v69 = *(_QWORD *)(v54 + 400);
          v74 = (_QWORD *)(v69 + 8 * v73);
          v75 = (_QWORD *)(v69 + 8 * v56);
          v76 = *(_QWORD *)(v54 + 416);
          if ( v74 != v75 )
          {
            do
              *v75++ = v76;
            while ( v74 != v75 );
            v69 = *(_QWORD *)(v54 + 400);
          }
          *(_DWORD *)(v54 + 408) = v68;
          goto LABEL_73;
        }
      }
      else
      {
        *(_DWORD *)(v54 + 408) = v68;
      }
    }
    v69 = *(_QWORD *)(v54 + 400);
LABEL_73:
    v91 = (_QWORD *)v54;
    *(_QWORD *)(v69 + 8LL * (v55 & 0x7FFFFFFF)) = sub_1DBA290(v55);
    v70 = v91;
    v92 = *(_QWORD *)(v91[50] + 8 * v58);
    sub_1DBB110(v70, v92);
    v54 = v53[2];
    v59 = v92;
    goto LABEL_58;
  }
LABEL_83:
  sub_2100870(v48, *(_QWORD *)(*(_QWORD *)(v6 + 24) + 256LL), *(_QWORD *)(*(_QWORD *)v6 + 24LL), *(_QWORD *)(v6 + 64));
  if ( (_BYTE *)v100[0] != v101 )
    _libc_free(v100[0]);
}
