// Function: sub_35BBA80
// Address: 0x35bba80
//
void __fastcall sub_35BBA80(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r14
  int v11; // r9d
  unsigned __int64 v12; // rdx
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // r9
  __int64 (*v19)(); // rdx
  __int64 v20; // rax
  unsigned int v21; // eax
  __int64 (*v22)(); // rax
  __int64 v23; // r8
  int *v24; // r13
  __int64 v25; // rdx
  __int64 v26; // r9
  unsigned __int64 v27; // rcx
  unsigned int v28; // eax
  __int64 v29; // r12
  unsigned int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rbx
  bool v38; // zf
  _QWORD **v39; // rsi
  _QWORD **v40; // rdx
  _QWORD **v41; // rax
  __int64 v42; // rcx
  __int64 v43; // rax
  unsigned __int64 v44; // r10
  _QWORD *v45; // rdx
  _QWORD *v46; // rdi
  unsigned __int64 v47; // rsi
  __int64 *v48; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // r11
  _QWORD *v51; // rcx
  _QWORD *v52; // rdi
  unsigned __int64 v53; // [rsp+0h] [rbp-140h]
  unsigned __int64 v54; // [rsp+0h] [rbp-140h]
  __int64 v55; // [rsp+8h] [rbp-138h]
  __int64 v56; // [rsp+10h] [rbp-130h]
  unsigned int v57; // [rsp+10h] [rbp-130h]
  __int64 v58; // [rsp+18h] [rbp-128h]
  __int64 v59; // [rsp+18h] [rbp-128h]
  __int64 v61; // [rsp+20h] [rbp-120h]
  __int64 v62; // [rsp+20h] [rbp-120h]
  __int64 v63; // [rsp+20h] [rbp-120h]
  int v64; // [rsp+28h] [rbp-118h]
  unsigned int v65[4]; // [rsp+2Ch] [rbp-114h] BYREF
  unsigned int v66; // [rsp+3Ch] [rbp-104h] BYREF
  _QWORD v67[2]; // [rsp+40h] [rbp-100h] BYREF
  __int64 v68; // [rsp+50h] [rbp-F0h]
  __int64 v69; // [rsp+58h] [rbp-E8h]
  __int64 v70; // [rsp+60h] [rbp-E0h]
  __int64 v71; // [rsp+68h] [rbp-D8h]
  __int64 v72; // [rsp+70h] [rbp-D0h]
  __int64 v73; // [rsp+78h] [rbp-C8h]
  unsigned int v74; // [rsp+80h] [rbp-C0h]
  char v75; // [rsp+84h] [rbp-BCh]
  __int64 v76; // [rsp+88h] [rbp-B8h]
  __int64 v77; // [rsp+90h] [rbp-B0h]
  char *v78; // [rsp+98h] [rbp-A8h]
  __int64 v79; // [rsp+A0h] [rbp-A0h]
  int v80; // [rsp+A8h] [rbp-98h]
  char v81; // [rsp+ACh] [rbp-94h]
  char v82; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v83; // [rsp+D0h] [rbp-70h]
  char *v84; // [rsp+D8h] [rbp-68h]
  __int64 v85; // [rsp+E0h] [rbp-60h]
  int v86; // [rsp+E8h] [rbp-58h]
  char v87; // [rsp+ECh] [rbp-54h]
  char v88; // [rsp+F0h] [rbp-50h] BYREF

  v7 = a1 + 208;
  v65[0] = a2;
  sub_35B8930((_QWORD *)(a1 + 208), v65);
  v11 = v65[0];
  v12 = *(unsigned int *)(a5 + 160);
  v13 = a1 + 304;
  v14 = v65[0] & 0x7FFFFFFF;
  v15 = 8LL * (v65[0] & 0x7FFFFFFF);
  if ( (v65[0] & 0x7FFFFFFF) >= (unsigned int)v12
    || (v15 = 8LL * v14, (v16 = *(_QWORD *)(*(_QWORD *)(a5 + 152) + v15)) == 0) )
  {
    v33 = v14 + 1;
    if ( (unsigned int)v12 < v33 )
    {
      v47 = v33;
      if ( v33 != v12 )
      {
        if ( v33 >= v12 )
        {
          v49 = *(_QWORD *)(a5 + 168);
          v50 = v47 - v12;
          if ( v47 > *(unsigned int *)(a5 + 164) )
          {
            v54 = v47 - v12;
            v55 = *(_QWORD *)(a5 + 168);
            v57 = v65[0];
            sub_C8D5F0(a5 + 152, (const void *)(a5 + 168), v47, 8u, v13, v65[0]);
            v12 = *(unsigned int *)(a5 + 160);
            v50 = v54;
            v49 = v55;
            v11 = v57;
            v13 = a1 + 304;
          }
          v34 = *(_QWORD *)(a5 + 152);
          v51 = (_QWORD *)(v34 + 8 * v12);
          v52 = &v51[v50];
          if ( v51 != v52 )
          {
            do
              *v51++ = v49;
            while ( v52 != v51 );
            LODWORD(v12) = *(_DWORD *)(a5 + 160);
            v34 = *(_QWORD *)(a5 + 152);
          }
          *(_DWORD *)(a5 + 160) = v50 + v12;
          goto LABEL_20;
        }
        *(_DWORD *)(a5 + 160) = v33;
      }
    }
    v34 = *(_QWORD *)(a5 + 152);
LABEL_20:
    v56 = v13;
    v35 = sub_2E10F30(v11);
    *(_QWORD *)(v34 + v15) = v35;
    v59 = v35;
    sub_2E11E80((_QWORD *)a5, v35);
    v13 = v56;
    v16 = v59;
  }
  v17 = *(_QWORD *)(a4 + 32);
  v67[1] = v16;
  v18 = *(_QWORD *)(a4 + 16);
  v68 = a3;
  v69 = v17;
  v67[0] = &unk_4A388F0;
  v70 = a5;
  v71 = a6;
  v19 = *(__int64 (**)())(*(_QWORD *)v18 + 128LL);
  v20 = 0;
  if ( v19 != sub_2DAC790 )
  {
    v62 = v13;
    v20 = ((__int64 (__fastcall *)(__int64))v19)(v18);
    v17 = v69;
    v13 = v62;
  }
  v72 = v20;
  v21 = *(_DWORD *)(a3 + 8);
  v73 = 0;
  v74 = v21;
  v78 = &v82;
  v75 = 0;
  v76 = v13;
  v77 = 0;
  v79 = 4;
  v80 = 0;
  v81 = 1;
  v83 = 0;
  v84 = &v88;
  v85 = 4;
  v86 = 0;
  v87 = 1;
  if ( !*(_BYTE *)(v17 + 36) )
    goto LABEL_41;
  v22 = *(__int64 (**)())(v17 + 16);
  v16 = *(unsigned int *)(v17 + 28);
  v19 = (__int64 (*)())((char *)v22 + 8 * v16);
  if ( v22 == v19 )
  {
LABEL_40:
    if ( (unsigned int)v16 >= *(_DWORD *)(v17 + 24) )
    {
LABEL_41:
      sub_C8CC70(v17 + 8, (__int64)v67, (__int64)v19, v16, v13, v18);
      goto LABEL_10;
    }
    *(_DWORD *)(v17 + 28) = v16 + 1;
    *(_QWORD *)v19 = v67;
    ++*(_QWORD *)(v17 + 8);
  }
  else
  {
    while ( *(_QWORD **)v22 != v67 )
    {
      v22 = (__int64 (*)())((char *)v22 + 8);
      if ( v19 == v22 )
        goto LABEL_40;
    }
  }
LABEL_10:
  (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a7 + 24LL))(a7, v67);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a4 + 16) + 200LL))(*(_QWORD *)(a4 + 16));
  v58 = *(_QWORD *)v68 + 4LL * *(unsigned int *)(v68 + 8);
  if ( *(_QWORD *)v68 + 4LL * v74 != v58 )
  {
    v24 = (int *)(*(_QWORD *)v68 + 4LL * v74);
    do
    {
      v26 = (unsigned int)*v24;
      v27 = *(unsigned int *)(a5 + 160);
      v28 = *v24 & 0x7FFFFFFF;
      v29 = 8LL * v28;
      if ( v28 < (unsigned int)v27 )
      {
        v25 = *(_QWORD *)(*(_QWORD *)(a5 + 152) + 8LL * v28);
        if ( v25 )
          goto LABEL_13;
      }
      v30 = v28 + 1;
      if ( (unsigned int)v27 < v30 )
      {
        v36 = v30;
        if ( v30 != v27 )
        {
          if ( v30 >= v27 )
          {
            v43 = *(_QWORD *)(a5 + 168);
            v44 = v36 - v27;
            if ( v36 > *(unsigned int *)(a5 + 164) )
            {
              v64 = *v24;
              v53 = v36 - v27;
              v63 = *(_QWORD *)(a5 + 168);
              sub_C8D5F0(a5 + 152, (const void *)(a5 + 168), v36, 8u, v23, v26);
              v27 = *(unsigned int *)(a5 + 160);
              LODWORD(v26) = v64;
              v44 = v53;
              v43 = v63;
            }
            v31 = *(_QWORD *)(a5 + 152);
            v45 = (_QWORD *)(v31 + 8 * v27);
            v46 = &v45[v44];
            if ( v45 != v46 )
            {
              do
                *v45++ = v43;
              while ( v46 != v45 );
              LODWORD(v27) = *(_DWORD *)(a5 + 160);
              v31 = *(_QWORD *)(a5 + 152);
            }
            *(_DWORD *)(a5 + 160) = v44 + v27;
            goto LABEL_17;
          }
          *(_DWORD *)(a5 + 160) = v30;
        }
      }
      v31 = *(_QWORD *)(a5 + 152);
LABEL_17:
      v32 = sub_2E10F30(v26);
      *(_QWORD *)(v31 + v29) = v32;
      v61 = v32;
      sub_2E11E80((_QWORD *)a5, v32);
      v25 = v61;
LABEL_13:
      ++v24;
      v66 = *(_DWORD *)(v25 + 112);
      sub_2DCBE50(v7, &v66);
    }
    while ( (int *)v58 != v24 );
  }
  v37 = v69;
  v38 = *(_BYTE *)(v69 + 36) == 0;
  v67[0] = &unk_4A388F0;
  if ( v38 )
  {
    v48 = sub_C8CA60(v69 + 8, (__int64)v67);
    if ( v48 )
    {
      *v48 = -2;
      ++*(_DWORD *)(v37 + 32);
      ++*(_QWORD *)(v37 + 8);
    }
  }
  else
  {
    v39 = *(_QWORD ***)(v69 + 16);
    v40 = &v39[*(unsigned int *)(v69 + 28)];
    v41 = v39;
    if ( v39 != v40 )
    {
      while ( *v41 != v67 )
      {
        if ( v40 == ++v41 )
          goto LABEL_30;
      }
      v42 = (unsigned int)(*(_DWORD *)(v69 + 28) - 1);
      *(_DWORD *)(v69 + 28) = v42;
      *v41 = v39[v42];
      ++*(_QWORD *)(v37 + 8);
    }
  }
LABEL_30:
  if ( !v87 )
    _libc_free((unsigned __int64)v84);
  if ( !v81 )
    _libc_free((unsigned __int64)v78);
}
