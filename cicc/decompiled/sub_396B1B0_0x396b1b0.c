// Function: sub_396B1B0
// Address: 0x396b1b0
//
__int64 __fastcall sub_396B1B0(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // r12
  double v36; // xmm4_8
  double v37; // xmm5_8
  unsigned int v38; // r14d
  _QWORD *v39; // rbx
  _QWORD *v40; // r12
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  _QWORD *v43; // rbx
  _QWORD *v44; // r12
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  unsigned __int64 v47; // r12
  unsigned __int64 v48; // rdi
  __int64 v49; // rax
  _QWORD *v50; // rbx
  _QWORD *v51; // r13
  unsigned __int64 v52; // r15
  unsigned __int64 v53; // rdi
  __int64 v55; // [rsp+8h] [rbp-178h]
  _QWORD v56[6]; // [rsp+10h] [rbp-170h] BYREF
  int v57; // [rsp+40h] [rbp-140h]
  unsigned __int64 v58; // [rsp+48h] [rbp-138h]
  int v59; // [rsp+50h] [rbp-130h]
  __int16 v60; // [rsp+54h] [rbp-12Ch]
  char v61; // [rsp+56h] [rbp-12Ah]
  int v62; // [rsp+58h] [rbp-128h]
  __int64 v63; // [rsp+60h] [rbp-120h]
  unsigned __int64 v64; // [rsp+68h] [rbp-118h]
  __int64 v65; // [rsp+70h] [rbp-110h]
  int v66; // [rsp+78h] [rbp-108h]
  _QWORD v67[3]; // [rsp+80h] [rbp-100h] BYREF
  _QWORD v68[4]; // [rsp+98h] [rbp-E8h] BYREF
  _BYTE *v69; // [rsp+B8h] [rbp-C8h]
  _BYTE *v70; // [rsp+C0h] [rbp-C0h]
  __int64 v71; // [rsp+C8h] [rbp-B8h]
  int v72; // [rsp+D0h] [rbp-B0h]
  _BYTE v73[64]; // [rsp+D8h] [rbp-A8h] BYREF
  __int64 v74; // [rsp+118h] [rbp-68h]
  unsigned __int64 v75; // [rsp+120h] [rbp-60h]
  __int64 v76; // [rsp+128h] [rbp-58h]
  int v77; // [rsp+130h] [rbp-50h]
  unsigned __int64 v78; // [rsp+138h] [rbp-48h]
  __int64 v79; // [rsp+140h] [rbp-40h]
  __int64 v80; // [rsp+148h] [rbp-38h]

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_64:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4FB9E2C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_64;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4FB9E2C);
  v14 = *(__int64 **)(a1 + 8);
  v15 = *(_DWORD *)(v13 + 164);
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_60:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9D3C0 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_60;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9D3C0);
  v19 = sub_14A4050(v18, a2);
  v20 = *(__int64 **)(a1 + 8);
  v21 = v19;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_61:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F97E48 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_61;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F97E48);
  v25 = *(__int64 **)(a1 + 8);
  v55 = v24 + 160;
  v26 = *v25;
  v27 = v25[1];
  if ( v26 == v27 )
LABEL_62:
    BUG();
  while ( *(_UNKNOWN **)v26 != &unk_4F9E06C )
  {
    v26 += 16;
    if ( v27 == v26 )
      goto LABEL_62;
  }
  v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(*(_QWORD *)(v26 + 8), &unk_4F9E06C);
  v29 = *(__int64 **)(a1 + 8);
  v30 = v28 + 160;
  v31 = *v29;
  v32 = v29[1];
  if ( v31 == v32 )
LABEL_63:
    BUG();
  while ( *(_UNKNOWN **)v31 != &unk_4F9920C )
  {
    v31 += 16;
    if ( v32 == v31 )
      goto LABEL_63;
  }
  v33 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(*(_QWORD *)(v31 + 8), &unk_4F9920C);
  v34 = *(_QWORD *)(a2 + 40);
  v56[0] = a2;
  v35 = v33;
  v56[5] = v21;
  v56[1] = sub_1632FA0(v34);
  v57 = v15;
  v56[4] = v55;
  v60 = 256;
  v56[3] = v30;
  v69 = v73;
  v70 = v73;
  v56[2] = v35 + 160;
  v58 = 0;
  v59 = 5;
  v61 = 0;
  v62 = 30;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67[1] = v67;
  v67[0] = v67;
  v67[2] = 0;
  v68[1] = v68;
  v68[0] = v68;
  v68[2] = 0;
  v68[3] = 0;
  v71 = 8;
  v72 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v38 = sub_396A6C0((__int64)v56, a3, a4, a5, a6, v36, v37, a9, a10);
  if ( v78 )
    j_j___libc_free_0(v78);
  j___libc_free_0(v75);
  if ( v70 != v69 )
    _libc_free((unsigned __int64)v70);
  v39 = (_QWORD *)v68[0];
  while ( v39 != v68 )
  {
    v40 = v39;
    v39 = (_QWORD *)*v39;
    v41 = v40[20];
    if ( (_QWORD *)v41 != v40 + 22 )
      _libc_free(v41);
    v42 = v40[9];
    if ( v42 != v40[8] )
      _libc_free(v42);
    j_j___libc_free_0((unsigned __int64)v40);
  }
  v43 = (_QWORD *)v67[0];
  while ( v43 != v67 )
  {
    v44 = v43;
    v43 = (_QWORD *)*v43;
    v45 = v44[20];
    if ( v45 != v44[19] )
      _libc_free(v45);
    v46 = v44[8];
    if ( (_QWORD *)v46 != v44 + 10 )
      _libc_free(v46);
    j_j___libc_free_0((unsigned __int64)v44);
  }
  j___libc_free_0(v64);
  v47 = v58;
  if ( v58 )
  {
    v48 = *(_QWORD *)(v58 + 192);
    if ( v48 != *(_QWORD *)(v58 + 184) )
      _libc_free(v48);
    j___libc_free_0(*(_QWORD *)(v47 + 152));
    v49 = *(unsigned int *)(v47 + 136);
    if ( (_DWORD)v49 )
    {
      v50 = *(_QWORD **)(v47 + 120);
      v51 = &v50[2 * v49];
      do
      {
        if ( *v50 != -16 && *v50 != -8 )
        {
          v52 = v50[1];
          if ( v52 )
          {
            _libc_free(*(_QWORD *)(v52 + 48));
            _libc_free(*(_QWORD *)(v52 + 24));
            j_j___libc_free_0(v52);
          }
        }
        v50 += 2;
      }
      while ( v51 != v50 );
    }
    j___libc_free_0(*(_QWORD *)(v47 + 120));
    v53 = *(_QWORD *)(v47 + 88);
    if ( v53 )
      j_j___libc_free_0(v53);
    j___libc_free_0(*(_QWORD *)(v47 + 64));
    j_j___libc_free_0(v47);
  }
  return v38;
}
