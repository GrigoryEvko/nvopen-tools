// Function: sub_38F7A10
// Address: 0x38f7a10
//
__int64 __fastcall sub_38F7A10(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned int a4)
{
  _QWORD *v4; // r15
  __int64 v5; // r13
  __int64 v6; // r12
  unsigned __int8 v7; // al
  unsigned int i; // r14d
  unsigned __int64 *v9; // r13
  unsigned __int64 *v10; // r15
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rbx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  __int64 v18; // rcx
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int64 v23; // r14
  __int64 *v24; // r13
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // eax
  unsigned __int64 v32; // r8
  unsigned __int64 v33; // r13
  unsigned __int64 v34; // rbx
  __int64 v35; // r15
  __int64 v36; // r12
  unsigned __int64 v37; // rdi
  __int64 v38; // r12
  unsigned __int64 v39; // rbx
  unsigned __int64 v40; // rdi
  _QWORD *v41; // [rsp+8h] [rbp-228h]
  unsigned __int64 v42; // [rsp+10h] [rbp-220h]
  __int64 *v43; // [rsp+10h] [rbp-220h]
  __int64 v44; // [rsp+30h] [rbp-200h]
  __int64 v45; // [rsp+38h] [rbp-1F8h]
  unsigned __int64 *v46; // [rsp+50h] [rbp-1E0h] BYREF
  unsigned __int64 *v47; // [rsp+58h] [rbp-1D8h]
  __int64 v48; // [rsp+60h] [rbp-1D0h]
  unsigned __int64 v49; // [rsp+70h] [rbp-1C0h] BYREF
  __int64 v50; // [rsp+78h] [rbp-1B8h]
  __int64 v51; // [rsp+80h] [rbp-1B0h]
  __int64 v52[2]; // [rsp+90h] [rbp-1A0h] BYREF
  unsigned __int64 v53; // [rsp+A0h] [rbp-190h]
  __int64 v54; // [rsp+A8h] [rbp-188h]
  __int64 v55; // [rsp+B0h] [rbp-180h]
  __int16 v56; // [rsp+B8h] [rbp-178h]
  _QWORD v57[2]; // [rsp+C0h] [rbp-170h] BYREF
  __int64 v58; // [rsp+D0h] [rbp-160h]
  __int64 v59; // [rsp+D8h] [rbp-158h]
  int v60; // [rsp+E0h] [rbp-150h]
  unsigned __int64 *v61; // [rsp+E8h] [rbp-148h]
  unsigned __int64 v62[2]; // [rsp+F0h] [rbp-140h] BYREF
  char v63; // [rsp+100h] [rbp-130h] BYREF
  char v64; // [rsp+101h] [rbp-12Fh]

  v4 = v57;
  v5 = a2;
  v6 = a1;
  v56 = 0;
  v52[0] = 0;
  v52[1] = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v57[0] = "expected identifier in '.irpc' directive";
  LOWORD(v58) = 259;
  v7 = sub_38F0EE0(a1, v52, a3, a4);
  if ( (unsigned __int8)sub_3909CB0(a1, v7, v57) )
    goto LABEL_2;
  v64 = 1;
  v62[0] = (unsigned __int64)"expected comma in '.irpc' directive";
  v63 = 3;
  if ( (unsigned __int8)sub_3909E20(a1, 25, v62) || (unsigned __int8)sub_38F6810(a1, 0, (__int64 *)&v46, v18, v19, v20) )
    goto LABEL_2;
  if ( (char *)v47 - (char *)v46 != 24 || v46[1] - *v46 != 40 )
  {
    v64 = 1;
    v62[0] = (unsigned __int64)"unexpected token in '.irpc' directive";
    v63 = 3;
    i = sub_3909CF0(a1, v62, 0, 0, v21, v22);
    goto LABEL_3;
  }
  v64 = 1;
  v62[0] = (unsigned __int64)"expected end of statement";
  v63 = 3;
  if ( (unsigned __int8)sub_3909E20(a1, 9, v62) || (v45 = sub_38EFA80(a1, a2)) == 0 )
  {
LABEL_2:
    i = 1;
    goto LABEL_3;
  }
  v62[0] = (unsigned __int64)&v63;
  v62[1] = 0x10000000000LL;
  v60 = 1;
  v59 = 0;
  v57[0] = &unk_49EFC48;
  v58 = 0;
  v57[1] = 0;
  v61 = v62;
  sub_16E7A40((__int64)v57, 0, 0, 0);
  v44 = *(_QWORD *)(*v46 + 8);
  if ( *(_QWORD *)(*v46 + 16) )
  {
    v23 = *(_QWORD *)(*v46 + 16);
    v24 = v52;
    v25 = 0;
    while ( 1 )
    {
      v42 = v25++;
      v49 = 0;
      v50 = 0;
      v51 = 0;
      v26 = sub_22077B0(0x28u);
      if ( v26 )
      {
        v27 = v42;
        *(_DWORD *)v26 = 2;
        *(_DWORD *)(v26 + 32) = 64;
        *(_QWORD *)(v26 + 24) = 0;
        if ( v23 <= v42 )
          v27 = v23;
        *(_QWORD *)(v26 + 8) = v27 + v44;
        v28 = v27;
        if ( v25 >= v27 )
          v28 = v25;
        if ( v28 > v23 )
          v28 = v23;
        *(_QWORD *)(v26 + 16) = v28 - v27;
      }
      v49 = v26;
      v50 = v26 + 40;
      v51 = v26 + 40;
      v29 = sub_3909460(v6);
      v30 = sub_39092A0(v29);
      v31 = sub_38E48B0(
              v6,
              (__int64)v4,
              *(unsigned __int8 **)(v45 + 16),
              *(_QWORD *)(v45 + 24),
              (__int64)v24,
              1,
              (__int64)&v49,
              1u,
              1,
              v30);
      if ( (_BYTE)v31 )
        break;
      v32 = v49;
      if ( v50 != v49 )
      {
        v43 = v24;
        v33 = v25;
        v34 = v49;
        v41 = v4;
        v35 = v6;
        v36 = v50;
        do
        {
          if ( *(_DWORD *)(v34 + 32) > 0x40u )
          {
            v37 = *(_QWORD *)(v34 + 24);
            if ( v37 )
              j_j___libc_free_0_0(v37);
          }
          v34 += 40LL;
        }
        while ( v36 != v34 );
        v25 = v33;
        v6 = v35;
        v24 = v43;
        v4 = v41;
        v32 = v49;
      }
      if ( v32 )
        j_j___libc_free_0(v32);
      if ( v23 == v25 )
      {
        v5 = a2;
        goto LABEL_54;
      }
    }
    v38 = v50;
    v39 = v49;
    for ( i = v31; v38 != v39; v39 += 40LL )
    {
      if ( *(_DWORD *)(v39 + 32) > 0x40u )
      {
        v40 = *(_QWORD *)(v39 + 24);
        if ( v40 )
          j_j___libc_free_0_0(v40);
      }
    }
    if ( v49 )
      j_j___libc_free_0(v49);
  }
  else
  {
LABEL_54:
    i = 0;
    sub_38EF860(v6, v5, v4);
  }
  v57[0] = &unk_49EFD28;
  sub_16E7960((__int64)v4);
  if ( (char *)v62[0] != &v63 )
    _libc_free(v62[0]);
LABEL_3:
  v9 = v47;
  v10 = v46;
  if ( v47 != v46 )
  {
    do
    {
      v11 = v10[1];
      v12 = *v10;
      if ( v11 != *v10 )
      {
        do
        {
          if ( *(_DWORD *)(v12 + 32) > 0x40u )
          {
            v13 = *(_QWORD *)(v12 + 24);
            if ( v13 )
              j_j___libc_free_0_0(v13);
          }
          v12 += 40LL;
        }
        while ( v11 != v12 );
        v12 = *v10;
      }
      if ( v12 )
        j_j___libc_free_0(v12);
      v10 += 3;
    }
    while ( v9 != v10 );
    v10 = v46;
  }
  if ( v10 )
    j_j___libc_free_0((unsigned __int64)v10);
  v14 = v54;
  v15 = v53;
  if ( v54 != v53 )
  {
    do
    {
      if ( *(_DWORD *)(v15 + 32) > 0x40u )
      {
        v16 = *(_QWORD *)(v15 + 24);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      v15 += 40LL;
    }
    while ( v14 != v15 );
    v15 = v53;
  }
  if ( v15 )
    j_j___libc_free_0(v15);
  return i;
}
