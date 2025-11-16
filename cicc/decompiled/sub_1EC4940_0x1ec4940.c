// Function: sub_1EC4940
// Address: 0x1ec4940
//
__int64 __fastcall sub_1EC4940(__int64 a1, int a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // esi
  unsigned int v8; // r8d
  __int64 v10; // rdi
  int v11; // ebx
  unsigned int v12; // ecx
  _DWORD *v13; // rax
  int v14; // edx
  bool v15; // al
  __int64 v16; // r8
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r12
  unsigned __int16 *v21; // rbx
  __int64 v22; // rax
  int v23; // r13d
  float v24; // xmm1_4
  unsigned int v25; // r13d
  __int64 v26; // rbx
  __int64 *v27; // rdx
  __int64 v28; // r13
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r13
  __int64 v33; // rbx
  __int64 *v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rax
  int v40; // r8d
  int v41; // r9d
  __int64 v42; // rdx
  __int64 v43; // r12
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // r14
  unsigned __int64 v47; // r13
  unsigned __int64 v48; // rax
  __int64 v49; // r15
  float v50; // xmm0_4
  int v51; // r12d
  unsigned int v52; // r14d
  int i; // r9d
  int v54; // r14d
  _DWORD *v55; // r9
  __int64 v56; // rdi
  int v57; // eax
  int v58; // edx
  unsigned int v60; // r12d
  __int64 v61; // rbx
  __int64 v62; // rdx
  __int64 v63; // rcx
  _QWORD *v64; // rdx
  _QWORD *v65; // rax
  unsigned int v66; // r15d
  __int64 v67; // rbx
  int v68; // eax
  int v69; // eax
  __int64 v70; // rsi
  unsigned int v71; // ebx
  int v72; // ecx
  int v73; // r8d
  _DWORD *v74; // rdi
  int v75; // eax
  int v76; // eax
  __int64 v77; // rsi
  int v78; // r8d
  unsigned int v79; // ebx
  int v80; // ecx
  __int64 v81; // rdx
  __int64 v82; // rcx
  _QWORD *v83; // rdx
  _QWORD *v84; // rax
  int *v85; // r11
  __int64 v86; // [rsp+0h] [rbp-D0h]
  unsigned int v87; // [rsp+Ch] [rbp-C4h]
  __int64 v88; // [rsp+10h] [rbp-C0h]
  int v89; // [rsp+18h] [rbp-B8h]
  int v90; // [rsp+1Ch] [rbp-B4h]
  __int64 v94; // [rsp+30h] [rbp-A0h]
  __int64 v95; // [rsp+30h] [rbp-A0h]
  unsigned __int16 *v96; // [rsp+38h] [rbp-98h]
  _QWORD *v97; // [rsp+38h] [rbp-98h]
  __int64 v98; // [rsp+40h] [rbp-90h]
  __int64 v99; // [rsp+48h] [rbp-88h]
  _QWORD v100[6]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v101; // [rsp+80h] [rbp-50h]
  __int64 v102; // [rsp+88h] [rbp-48h]
  int v103; // [rsp+90h] [rbp-40h]
  float (__fastcall *v104)(int, float); // [rsp+98h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 976);
  if ( !v6 )
  {
LABEL_42:
    LODWORD(v20) = 0;
    return (unsigned int)v20;
  }
  v8 = v6 - 1;
  v10 = *(_QWORD *)(a1 + 960);
  v11 = 37 * a2;
  v12 = (v6 - 1) & (37 * a2);
  v13 = (_DWORD *)(v10 + 12LL * v12);
  v14 = *v13;
  if ( a2 == *v13 )
  {
    v90 = v13[2];
    v89 = v13[1];
    v15 = v89 == 0 || v90 == 0;
    goto LABEL_4;
  }
  v51 = *v13;
  v52 = (v6 - 1) & (37 * a2);
  for ( i = 1; ; ++i )
  {
    if ( v51 == -1 )
      goto LABEL_42;
    v52 = v8 & (i + v52);
    v51 = *(_DWORD *)(v10 + 12LL * v52);
    if ( a2 == v51 )
      break;
  }
  v54 = 1;
  v55 = 0;
  while ( 1 )
  {
    if ( v14 == -1 )
    {
      v56 = a1 + 952;
      if ( !v55 )
        v55 = v13;
      v57 = *(_DWORD *)(a1 + 968);
      ++*(_QWORD *)(a1 + 952);
      v58 = v57 + 1;
      if ( 4 * (v57 + 1) >= 3 * v6 )
      {
        sub_168FE70(v56, 2 * v6);
        v68 = *(_DWORD *)(a1 + 976);
        if ( v68 )
        {
          v69 = v68 - 1;
          v70 = *(_QWORD *)(a1 + 960);
          v71 = v69 & v11;
          v55 = (_DWORD *)(v70 + 12LL * v71);
          v58 = *(_DWORD *)(a1 + 968) + 1;
          v72 = *v55;
          if ( v51 == *v55 )
            goto LABEL_39;
          v73 = 1;
          v74 = 0;
          while ( v72 != -1 )
          {
            if ( v72 == -2 && !v74 )
              v74 = v55;
            v71 = v69 & (v73 + v71);
            v55 = (_DWORD *)(v70 + 12LL * v71);
            v72 = *v55;
            if ( v51 == *v55 )
              goto LABEL_39;
            ++v73;
          }
LABEL_63:
          if ( v74 )
            v55 = v74;
          goto LABEL_39;
        }
      }
      else
      {
        if ( v6 - *(_DWORD *)(a1 + 972) - v58 > v6 >> 3 )
        {
LABEL_39:
          *(_DWORD *)(a1 + 968) = v58;
          if ( *v55 != -1 )
            --*(_DWORD *)(a1 + 972);
          *v55 = v51;
          *(_QWORD *)(v55 + 1) = 0;
          goto LABEL_42;
        }
        sub_168FE70(v56, v6);
        v75 = *(_DWORD *)(a1 + 976);
        if ( v75 )
        {
          v76 = v75 - 1;
          v77 = *(_QWORD *)(a1 + 960);
          v78 = 1;
          v79 = v76 & v11;
          v55 = (_DWORD *)(v77 + 12LL * v79);
          v58 = *(_DWORD *)(a1 + 968) + 1;
          v74 = 0;
          v80 = *v55;
          if ( v51 == *v55 )
            goto LABEL_39;
          while ( v80 != -1 )
          {
            if ( v80 == -2 && !v74 )
              v74 = v55;
            v79 = v76 & (v78 + v79);
            v55 = (_DWORD *)(v77 + 12LL * v79);
            v80 = *v55;
            if ( v51 == *v55 )
              goto LABEL_39;
            ++v78;
          }
          goto LABEL_63;
        }
      }
      ++*(_DWORD *)(a1 + 968);
      BUG();
    }
    if ( v14 != -2 || v55 )
      v13 = v55;
    a6 = (unsigned int)(v54 + 1);
    v12 = v8 & (v12 + v54);
    v85 = (int *)(v10 + 12LL * v12);
    v14 = *v85;
    if ( v51 == *v85 )
      break;
    ++v54;
    v55 = v13;
    v13 = (_DWORD *)(v10 + 12LL * v12);
  }
  v89 = v85[1];
  v90 = v85[2];
  v15 = v90 == 0 || v89 == 0;
LABEL_4:
  if ( v15 )
    goto LABEL_42;
  v16 = *(_QWORD *)(a1 + 264);
  v17 = *(_QWORD *)(a3 + 16);
  v98 = *(_QWORD *)(v17 + 16);
  v99 = *(_QWORD *)(v17 + 8);
  v18 = *(unsigned int *)(v16 + 408);
  v87 = a2 & 0x7FFFFFFF;
  v19 = 8LL * (a2 & 0x7FFFFFFF);
  v88 = a2 & 0x7FFFFFFF;
  v86 = v19;
  if ( (a2 & 0x7FFFFFFFu) >= (unsigned int)v18
    || (v20 = *(_QWORD *)(*(_QWORD *)(v16 + 400) + 8LL * (a2 & 0x7FFFFFFF))) == 0 )
  {
    v60 = v87 + 1;
    if ( (unsigned int)v18 < v87 + 1 )
    {
      v62 = v60;
      if ( v60 >= v18 )
      {
        if ( v60 > v18 )
        {
          if ( v60 > (unsigned __int64)*(unsigned int *)(v16 + 412) )
          {
            v95 = *(_QWORD *)(a1 + 264);
            sub_16CD150(v16 + 400, (const void *)(v16 + 416), v60, 8, v16, a6);
            v16 = v95;
            v62 = v60;
            v18 = *(unsigned int *)(v95 + 408);
          }
          v61 = *(_QWORD *)(v16 + 400);
          v63 = *(_QWORD *)(v16 + 416);
          v64 = (_QWORD *)(v61 + 8 * v62);
          v65 = (_QWORD *)(v61 + 8 * v18);
          if ( v64 != v65 )
          {
            do
              *v65++ = v63;
            while ( v64 != v65 );
            v61 = *(_QWORD *)(v16 + 400);
          }
          *(_DWORD *)(v16 + 408) = v60;
          goto LABEL_46;
        }
      }
      else
      {
        *(_DWORD *)(v16 + 408) = v60;
      }
    }
    v61 = *(_QWORD *)(v16 + 400);
LABEL_46:
    v97 = (_QWORD *)v16;
    *(_QWORD *)(v86 + v61) = sub_1DBA290(a2);
    v20 = *(_QWORD *)(v97[50] + 8 * v88);
    sub_1DBB110(v97, v20);
  }
  v21 = *(unsigned __int16 **)(a5 + 48);
  v22 = *(_QWORD *)(a5 + 56);
  LODWORD(v100[0]) = -1;
  v23 = 0;
  v24 = *(float *)(v20 + 116);
  v96 = &v21[v22];
  *((float *)v100 + 1) = v24;
  if ( v21 != v96 )
  {
    v94 = v20;
    LODWORD(v20) = 0;
    do
    {
      v25 = *v21;
      if ( (unsigned __int8)sub_1EBC970((_QWORD *)a1, v94, v25, v99, v98, (__int64)v100) )
        LODWORD(v20) = v25;
      ++v21;
    }
    while ( v96 != v21 );
    v24 = *((float *)v100 + 1);
    v23 = v20;
  }
  LOBYTE(v20) = v90 != v23 && *(_DWORD *)a3 != v90;
  if ( (_BYTE)v20 )
    goto LABEL_42;
  v26 = *(_QWORD *)(a3 + 8);
  v27 = &qword_4FCF930;
  if ( v26 )
  {
    v28 = 24LL * a4;
    v27 = (__int64 *)(v28 + *(_QWORD *)(v26 + 512));
    if ( *(_DWORD *)v27 != *(_DWORD *)(v26 + 4) )
    {
      sub_20F85B0(v26, a4, v27, v19, v16, a6);
      v27 = (__int64 *)(v28 + *(_QWORD *)(v26 + 512));
    }
  }
  *(_QWORD *)(a3 + 16) = v27;
  v29 = *(_QWORD *)(a1 + 264);
  v30 = v89 & 0x7FFFFFFF;
  if ( (unsigned int)v30 >= *(_DWORD *)(v29 + 408) )
    goto LABEL_42;
  v31 = *(_QWORD *)(v29 + 400);
  v32 = *(_QWORD *)(v31 + 8 * v30);
  if ( !v32 )
    goto LABEL_42;
  v33 = v27[1];
  v34 = (__int64 *)sub_1DB3C70(*(__int64 **)(v31 + 8 * v30), v33);
  if ( v34 == (__int64 *)(*(_QWORD *)v32 + 24LL * *(unsigned int *)(v32 + 8))
    || (*(_DWORD *)((*v34 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v34 >> 1) & 3) > (*(_DWORD *)((v33 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                          | (unsigned int)(v33 >> 1) & 3) )
  {
    return (unsigned int)v20;
  }
  v35 = *(__int64 **)(a1 + 8);
  v36 = *v35;
  v37 = v35[1];
  if ( v36 == v37 )
LABEL_98:
    BUG();
  while ( *(_UNKNOWN **)v36 != &unk_4FC6A0C )
  {
    v36 += 16;
    if ( v37 == v36 )
      goto LABEL_98;
  }
  v38 = *(_QWORD *)(a1 + 808);
  v39 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v36 + 8) + 104LL))(*(_QWORD *)(v36 + 8), &unk_4FC6A0C);
  v42 = *(_QWORD *)(a1 + 256);
  v43 = *(_QWORD *)(a1 + 264);
  v100[4] = v38;
  v100[3] = v39;
  v44 = *(_QWORD *)(a1 + 680);
  v104 = sub_1EBAF90;
  v100[2] = v42;
  v45 = *(_QWORD *)(a3 + 16);
  v100[0] = v44;
  v100[1] = v43;
  v100[5] = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v46 = *(_QWORD *)(v45 + 16);
  v47 = *(_QWORD *)(v45 + 8) & 6LL | *(_QWORD *)(*(_QWORD *)(v45 + 8) & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL;
  v48 = *(unsigned int *)(v43 + 408);
  if ( v87 >= (unsigned int)v48 || (v49 = *(_QWORD *)(*(_QWORD *)(v43 + 400) + 8 * v88)) == 0 )
  {
    v66 = v87 + 1;
    if ( (unsigned int)v48 < v87 + 1 )
    {
      v81 = v66;
      if ( v66 < v48 )
      {
        *(_DWORD *)(v43 + 408) = v66;
        v67 = *(_QWORD *)(v43 + 400);
        goto LABEL_58;
      }
      if ( v66 > v48 )
      {
        if ( v66 > (unsigned __int64)*(unsigned int *)(v43 + 412) )
        {
          sub_16CD150(v43 + 400, (const void *)(v43 + 416), v66, 8, v40, v41);
          v48 = *(unsigned int *)(v43 + 408);
          v81 = v66;
        }
        v67 = *(_QWORD *)(v43 + 400);
        v82 = *(_QWORD *)(v43 + 416);
        v83 = (_QWORD *)(v67 + 8 * v81);
        v84 = (_QWORD *)(v67 + 8 * v48);
        if ( v83 != v84 )
        {
          do
            *v84++ = v82;
          while ( v83 != v84 );
          v67 = *(_QWORD *)(v43 + 400);
        }
        *(_DWORD *)(v43 + 408) = v66;
        goto LABEL_58;
      }
    }
    v67 = *(_QWORD *)(v43 + 400);
LABEL_58:
    *(_QWORD *)(v86 + v67) = sub_1DBA290(a2);
    v49 = *(_QWORD *)(*(_QWORD *)(v43 + 400) + 8 * v88);
    sub_1DBB110((_QWORD *)v43, v49);
  }
  LODWORD(v20) = 1;
  v50 = sub_20E3050(v100, v49, v47, v46);
  if ( v50 >= 0.0 )
    LOBYTE(v20) = v24 <= v50;
  j___libc_free_0(v101);
  return (unsigned int)v20;
}
