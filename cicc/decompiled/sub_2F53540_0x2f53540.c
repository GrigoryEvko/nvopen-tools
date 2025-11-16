// Function: sub_2F53540
// Address: 0x2f53540
//
__int64 __fastcall sub_2F53540(__int64 a1, __int64 a2, unsigned int a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v10; // r8
  _QWORD *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // r8
  __int64 (*v16)(); // rsi
  __int64 v17; // rax
  int v18; // eax
  _QWORD **v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int64 v23; // r10
  int v24; // r11d
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rbx
  bool v28; // zf
  _QWORD **v29; // rsi
  _QWORD **v30; // rdx
  _QWORD **v31; // rax
  __int64 v32; // rcx
  unsigned __int64 v34; // r10
  int v35; // eax
  __int64 v36; // r10
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rbx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 *v44; // rax
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // [rsp+0h] [rbp-140h]
  int v47; // [rsp+0h] [rbp-140h]
  __int64 v48; // [rsp+0h] [rbp-140h]
  unsigned __int64 v49; // [rsp+0h] [rbp-140h]
  __int64 v50; // [rsp+0h] [rbp-140h]
  int v51; // [rsp+8h] [rbp-138h]
  int v52; // [rsp+8h] [rbp-138h]
  unsigned int *v54; // [rsp+10h] [rbp-130h] BYREF
  __int64 v55; // [rsp+18h] [rbp-128h]
  _BYTE v56[32]; // [rsp+20h] [rbp-120h] BYREF
  _QWORD v57[3]; // [rsp+40h] [rbp-100h] BYREF
  __int64 v58; // [rsp+58h] [rbp-E8h]
  __int64 v59; // [rsp+60h] [rbp-E0h]
  _QWORD *v60; // [rsp+68h] [rbp-D8h]
  __int64 v61; // [rsp+70h] [rbp-D0h]
  __int64 v62; // [rsp+78h] [rbp-C8h]
  int v63; // [rsp+80h] [rbp-C0h]
  char v64; // [rsp+84h] [rbp-BCh]
  __int64 v65; // [rsp+88h] [rbp-B8h]
  __int64 v66; // [rsp+90h] [rbp-B0h]
  char *v67; // [rsp+98h] [rbp-A8h]
  __int64 v68; // [rsp+A0h] [rbp-A0h]
  int v69; // [rsp+A8h] [rbp-98h]
  char v70; // [rsp+ACh] [rbp-94h]
  char v71; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v72; // [rsp+D0h] [rbp-70h]
  char *v73; // [rsp+D8h] [rbp-68h]
  __int64 v74; // [rsp+E0h] [rbp-60h]
  int v75; // [rsp+E8h] [rbp-58h]
  char v76; // [rsp+ECh] [rbp-54h]
  char v77; // [rsp+F0h] [rbp-50h] BYREF

  v7 = a1 + 400;
  v10 = *(_QWORD *)(a1 + 32);
  v11 = *(_QWORD **)(a1 + 24);
  v55 = 0x800000000LL;
  v12 = *(_QWORD *)(a1 + 768);
  v13 = a1 + 760;
  v57[2] = a5;
  v57[1] = a2;
  v54 = (unsigned int *)v56;
  v57[0] = &unk_4A388F0;
  v14 = *(_QWORD *)(v12 + 32);
  v59 = v10;
  v58 = v14;
  v60 = v11;
  v15 = *(_QWORD *)(v12 + 16);
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 128LL);
  v17 = 0;
  if ( v16 != sub_2DAC790 )
  {
    v50 = v13;
    v17 = ((__int64 (__fastcall *)(__int64))v16)(v15);
    v14 = v58;
    v13 = v50;
  }
  v61 = v17;
  v18 = *(_DWORD *)(a5 + 8);
  v65 = v7;
  v63 = v18;
  v67 = &v71;
  v62 = v13;
  v64 = 0;
  v66 = 0;
  v68 = 4;
  v69 = 0;
  v70 = 1;
  v72 = 0;
  v73 = &v77;
  v74 = 4;
  v75 = 0;
  v76 = 1;
  if ( !*(_BYTE *)(v14 + 36) )
    goto LABEL_29;
  v19 = *(_QWORD ***)(v14 + 16);
  v13 = *(unsigned int *)(v14 + 28);
  v11 = &v19[v13];
  if ( v19 == v11 )
  {
LABEL_28:
    if ( (unsigned int)v13 >= *(_DWORD *)(v14 + 24) )
    {
LABEL_29:
      sub_C8CC70(v14 + 8, (__int64)v57, (__int64)v11, v13, v15, a6);
      goto LABEL_8;
    }
    *(_DWORD *)(v14 + 28) = v13 + 1;
    *v11 = v57;
    ++*(_QWORD *)(v14 + 8);
  }
  else
  {
    while ( *v19 != v57 )
    {
      if ( v11 == ++v19 )
        goto LABEL_28;
    }
  }
LABEL_8:
  sub_2FB3410(*(_QWORD *)(a1 + 1000), v57, (unsigned int)dword_5024248);
  v22 = *(_QWORD *)(a1 + 824);
  v23 = *(unsigned int *)(v22 + 56);
  v24 = v23;
  if ( *(_DWORD *)(a1 + 28812) < (unsigned int)v23 )
  {
    v45 = *(unsigned int *)(v22 + 56);
    *(_DWORD *)(a1 + 28808) = 0;
    v52 = v23;
    v49 = v23;
    sub_C8D5F0(a1 + 28800, (const void *)(a1 + 28816), v45, 4u, v20, v21);
    memset(*(void **)(a1 + 28800), 255, 4 * v49);
    *(_DWORD *)(a1 + 28808) = v52;
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 28808);
    v26 = v25;
    if ( v23 <= v25 )
      v26 = v23;
    if ( v26 )
    {
      v51 = v23;
      v46 = v23;
      memset(*(void **)(a1 + 28800), 255, 4 * v26);
      v25 = *(unsigned int *)(a1 + 28808);
      v24 = v51;
      v23 = v46;
    }
    if ( v23 > v25 )
    {
      v34 = v23 - v25;
      if ( v34 )
      {
        if ( 4 * v34 )
        {
          v47 = v24;
          memset((void *)(*(_QWORD *)(a1 + 28800) + 4 * v25), 255, 4 * v34);
          v24 = v47;
        }
      }
    }
    *(_DWORD *)(a1 + 28808) = v24;
  }
  if ( a3 == -1
    || (v48 = *(_QWORD *)(a1 + 24176) + 144LL * a3, v35 = sub_2F4F510(v48, (_QWORD *)(a1 + 28800), a3), v36 = v48, !v35) )
  {
    if ( !a4 )
      goto LABEL_17;
    goto LABEL_37;
  }
  v37 = (unsigned int)v55;
  v38 = (unsigned int)v55 + 1LL;
  if ( v38 > HIDWORD(v55) )
  {
    sub_C8D5F0((__int64)&v54, v56, v38, 4u, v20, v21);
    v37 = (unsigned int)v55;
    v36 = v48;
  }
  v54[v37] = a3;
  v39 = *(_QWORD *)(a1 + 1000);
  LODWORD(v55) = v55 + 1;
  *(_DWORD *)(v36 + 4) = sub_2FB2500(v39);
  if ( a4 )
  {
LABEL_37:
    v40 = *(_QWORD *)(a1 + 24176);
    if ( (unsigned int)sub_2F4F510(v40, (_QWORD *)(a1 + 28800), 0) )
    {
      v41 = (unsigned int)v55;
      v42 = (unsigned int)v55 + 1LL;
      if ( v42 > HIDWORD(v55) )
      {
        sub_C8D5F0((__int64)&v54, v56, v42, 4u, v20, v21);
        v41 = (unsigned int)v55;
      }
      v54[v41] = 0;
      v43 = *(_QWORD *)(a1 + 1000);
      LODWORD(v55) = v55 + 1;
      *(_DWORD *)(v40 + 4) = sub_2FB2500(v43);
    }
  }
LABEL_17:
  sub_2F529A0(a1, (__int64)v57, v54, (unsigned int)v55, v20, v21);
  v27 = v58;
  v28 = *(_BYTE *)(v58 + 36) == 0;
  v57[0] = &unk_4A388F0;
  if ( !v28 )
  {
    v29 = *(_QWORD ***)(v58 + 16);
    v30 = &v29[*(unsigned int *)(v58 + 28)];
    v31 = v29;
    if ( v29 != v30 )
    {
      while ( *v31 != v57 )
      {
        if ( v30 == ++v31 )
          goto LABEL_23;
      }
      v32 = (unsigned int)(*(_DWORD *)(v58 + 28) - 1);
      *(_DWORD *)(v58 + 28) = v32;
      *v31 = v29[v32];
      ++*(_QWORD *)(v27 + 8);
    }
LABEL_23:
    if ( v76 )
      goto LABEL_24;
LABEL_43:
    _libc_free((unsigned __int64)v73);
    if ( v70 )
      goto LABEL_25;
    goto LABEL_44;
  }
  v44 = sub_C8CA60(v58 + 8, (__int64)v57);
  if ( !v44 )
    goto LABEL_23;
  *v44 = -2;
  ++*(_DWORD *)(v27 + 32);
  ++*(_QWORD *)(v27 + 8);
  if ( !v76 )
    goto LABEL_43;
LABEL_24:
  if ( v70 )
    goto LABEL_25;
LABEL_44:
  _libc_free((unsigned __int64)v67);
LABEL_25:
  if ( v54 != (unsigned int *)v56 )
    _libc_free((unsigned __int64)v54);
  return 0;
}
