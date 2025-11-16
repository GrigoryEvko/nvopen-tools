// Function: sub_2365080
// Address: 0x2365080
//
__int64 *__fastcall sub_2365080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v10; // rax
  __int64 *v11; // rax
  __int64 *v12; // rdx
  unsigned int v13; // esi
  __int64 v14; // r9
  int v15; // r11d
  __int64 **v16; // rdi
  unsigned int v17; // ecx
  _QWORD *v18; // rax
  __int64 *v19; // r8
  __int64 *v20; // rdx
  __int64 *v21; // rax
  __int64 *v22; // rbx
  __int64 *v23; // rbx
  __int64 *result; // rax
  __int64 *v25; // rbx
  __int64 *v26; // rbx
  __int64 *v27; // rbx
  __int64 *v28; // rbx
  __int64 *v29; // rbx
  __int64 v30; // rdi
  _QWORD *v31; // rax
  __int64 v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // rdi
  _QWORD *v35; // rax
  __int64 v36; // rdi
  _QWORD *v37; // rax
  __int64 v38; // rdi
  _QWORD *v39; // rax
  __int64 v40; // rdi
  _QWORD *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rdi
  _QWORD *v44; // rax
  __int64 v45; // rdi
  _QWORD *v46; // rax
  __int64 v47; // rdi
  _QWORD *v48; // rax
  __int64 v49; // rdi
  int v50; // eax
  int v51; // ecx
  __int64 *v52; // [rsp+0h] [rbp-50h]
  __int64 *v53; // [rsp+0h] [rbp-50h]
  __int64 *v54; // [rsp+0h] [rbp-50h]
  __int64 *v55; // [rsp+0h] [rbp-50h]
  __int64 *v57; // [rsp+10h] [rbp-40h] BYREF
  __int64 v58[7]; // [rsp+18h] [rbp-38h] BYREF

  v58[0] = (__int64)&unk_4F82418;
  v10 = sub_23624E0(a5, v58);
  if ( !*v10 )
  {
    v53 = v10;
    v33 = (_QWORD *)sub_22077B0(0x10u);
    if ( v33 )
    {
      v33[1] = a3;
      *v33 = &unk_4A156F8;
    }
    v34 = *v53;
    *v53 = (__int64)v33;
    if ( v34 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
  }
  v58[0] = (__int64)&unk_4FDADC0;
  v11 = sub_23624E0(a5, v58);
  if ( !*v11 )
  {
    v52 = v11;
    v31 = (_QWORD *)sub_22077B0(0x10u);
    if ( v31 )
    {
      v31[1] = a4;
      *v31 = &unk_4A15728;
    }
    v32 = *v52;
    *v52 = (__int64)v31;
    if ( v32 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
  }
  v12 = &qword_4FDADB8;
  v13 = *(_DWORD *)(a4 + 24);
  v57 = &qword_4FDADB8;
  if ( !v13 )
  {
    ++*(_QWORD *)a4;
    v58[0] = 0;
LABEL_75:
    v13 *= 2;
    goto LABEL_76;
  }
  v14 = *(_QWORD *)(a4 + 8);
  v15 = 1;
  v16 = 0;
  v17 = (v13 - 1) & (((unsigned int)&qword_4FDADB8 >> 9) ^ ((unsigned int)&qword_4FDADB8 >> 4));
  v18 = (_QWORD *)(v14 + 16LL * v17);
  v19 = (__int64 *)*v18;
  if ( (__int64 *)*v18 == &qword_4FDADB8 )
  {
LABEL_5:
    v20 = v18 + 1;
    goto LABEL_6;
  }
  while ( v19 != (__int64 *)-4096LL )
  {
    if ( v19 == (__int64 *)-8192LL && !v16 )
      v16 = (__int64 **)v18;
    v17 = (v13 - 1) & (v15 + v17);
    v18 = (_QWORD *)(v14 + 16LL * v17);
    v19 = (__int64 *)*v18;
    if ( (__int64 *)*v18 == &qword_4FDADB8 )
      goto LABEL_5;
    ++v15;
  }
  if ( !v16 )
    v16 = (__int64 **)v18;
  v50 = *(_DWORD *)(a4 + 16);
  ++*(_QWORD *)a4;
  v51 = v50 + 1;
  v58[0] = (__int64)v16;
  if ( 4 * (v50 + 1) >= 3 * v13 )
    goto LABEL_75;
  if ( v13 - *(_DWORD *)(a4 + 20) - v51 <= v13 >> 3 )
  {
LABEL_76:
    sub_2362DB0(a4, v13);
    sub_2351910(a4, (__int64 *)&v57, v58);
    v12 = v57;
    v16 = (__int64 **)v58[0];
    v51 = *(_DWORD *)(a4 + 16) + 1;
  }
  *(_DWORD *)(a4 + 16) = v51;
  if ( *v16 != (__int64 *)-4096LL )
    --*(_DWORD *)(a4 + 20);
  *v16 = v12;
  v20 = (__int64 *)(v16 + 1);
  v16[1] = 0;
LABEL_6:
  if ( !*v20 )
  {
    v55 = v20;
    v41 = (_QWORD *)sub_22077B0(0x10u);
    if ( v41 )
    {
      v41[1] = a5;
      *v41 = &unk_4A15758;
    }
    v42 = *v55;
    *v55 = (__int64)v41;
    if ( v42 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v42 + 8LL))(v42);
  }
  v58[0] = (__int64)&unk_4FDADB0;
  v21 = sub_2363490(a3, v58);
  if ( !*v21 )
  {
    v54 = v21;
    v39 = (_QWORD *)sub_22077B0(0x10u);
    if ( v39 )
    {
      v39[1] = a4;
      *v39 = &unk_4A15788;
    }
    v40 = *v54;
    *v54 = (__int64)v39;
    if ( v40 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v40 + 8LL))(v40);
  }
  v58[0] = (__int64)&unk_4F82410;
  v22 = sub_2363490(a3, v58);
  if ( !*v22 )
  {
    v37 = (_QWORD *)sub_22077B0(0x10u);
    if ( v37 )
    {
      v37[1] = a5;
      *v37 = &unk_4A157B8;
    }
    v38 = *v22;
    *v22 = (__int64)v37;
    if ( v38 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v38 + 8LL))(v38);
  }
  v58[0] = (__int64)&unk_4FDBCE0;
  v23 = sub_2363490(a3, v58);
  if ( !*v23 )
  {
    v35 = (_QWORD *)sub_22077B0(0x10u);
    if ( v35 )
    {
      v35[1] = a2;
      *v35 = &unk_4A157E8;
    }
    v36 = *v23;
    *v23 = (__int64)v35;
    if ( v36 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
  }
  v58[0] = (__int64)&unk_4FDBCD8;
  result = sub_2364F40(a2, v58);
  v25 = result;
  if ( !*result )
  {
    result = (__int64 *)sub_22077B0(0x10u);
    if ( result )
    {
      result[1] = a3;
      *result = (__int64)&unk_4A15818;
    }
    v30 = *v25;
    *v25 = (__int64)result;
    if ( v30 )
      result = (__int64 *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
  }
  if ( a6 )
  {
    v58[0] = (__int64)&unk_50209C8;
    v26 = sub_23624E0(a5, v58);
    if ( !*v26 )
    {
      v44 = (_QWORD *)sub_22077B0(0x10u);
      if ( v44 )
      {
        v44[1] = a6;
        *v44 = &unk_4A15848;
      }
      v45 = *v26;
      *v26 = (__int64)v44;
      if ( v45 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v45 + 8LL))(v45);
    }
    v58[0] = (__int64)&unk_50209C0;
    v27 = sub_2363490(a3, v58);
    if ( !*v27 )
    {
      v46 = (_QWORD *)sub_22077B0(0x10u);
      if ( v46 )
      {
        v46[1] = a6;
        *v46 = &unk_4A15878;
      }
      v47 = *v27;
      *v27 = (__int64)v46;
      if ( v47 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v47 + 8LL))(v47);
    }
    v58[0] = (__int64)&unk_50209B8;
    v28 = sub_2364C00(a6, v58);
    if ( !*v28 )
    {
      v48 = (_QWORD *)sub_22077B0(0x10u);
      if ( v48 )
      {
        v48[1] = a5;
        *v48 = &unk_4A158A8;
      }
      v49 = *v28;
      *v28 = (__int64)v48;
      if ( v49 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v49 + 8LL))(v49);
    }
    v58[0] = (__int64)&unk_50209D0;
    result = sub_2364C00(a6, v58);
    v29 = result;
    if ( !*result )
    {
      result = (__int64 *)sub_22077B0(0x10u);
      if ( result )
      {
        result[1] = a3;
        *result = (__int64)&unk_4A158D8;
      }
      v43 = *v29;
      *v29 = (__int64)result;
      if ( v43 )
        return (__int64 *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v43 + 8LL))(v43);
    }
  }
  return result;
}
