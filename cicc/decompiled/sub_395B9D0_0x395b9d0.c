// Function: sub_395B9D0
// Address: 0x395b9d0
//
__int64 __fastcall sub_395B9D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rdx
  _QWORD *v9; // r14
  __int64 v10; // rax
  __int64 v11; // r8
  int v12; // r9d
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r8
  int v16; // r9d
  __int64 v17; // rax
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // r13
  __int64 v21; // rax
  _BOOL8 v22; // r13
  __int64 v23; // rax
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // r13
  __int64 v27; // rax
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // r13
  _QWORD *v35; // rax
  __int64 v36; // r14
  int v37; // ecx
  bool v38; // zf
  __m128i v39; // xmm0
  int v41; // [rsp+0h] [rbp-100h]
  _BOOL8 v42; // [rsp+8h] [rbp-F8h]
  __int64 v43; // [rsp+8h] [rbp-F8h]
  __int64 v44; // [rsp+8h] [rbp-F8h]
  __int64 v45; // [rsp+8h] [rbp-F8h]
  __int64 *v48; // [rsp+28h] [rbp-D8h]
  __int64 *v49; // [rsp+28h] [rbp-D8h]
  __m128i v50; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v51; // [rsp+40h] [rbp-C0h]
  __int64 v52; // [rsp+58h] [rbp-A8h] BYREF
  const char *v53; // [rsp+60h] [rbp-A0h] BYREF
  char v54; // [rsp+70h] [rbp-90h]
  char v55; // [rsp+71h] [rbp-8Fh]
  __int64 *v56; // [rsp+80h] [rbp-80h] BYREF
  __int64 v57; // [rsp+88h] [rbp-78h]
  _QWORD v58[14]; // [rsp+90h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a6 + 104);
  v9 = **(_QWORD ***)(a5 + 40);
  v48 = *(__int64 **)(a5 + 40);
  v52 = v8;
  v10 = *(_QWORD *)(a7 + 16);
  if ( v10 )
  {
    v58[0] = v8;
    v58[1] = v10;
    v56 = v58;
    v57 = 0x200000002LL;
    v52 = sub_3958EE0(a3, (__int64 *)&v56, a4);
    if ( v56 != v58 )
      _libc_free((unsigned __int64)v56);
  }
  v56 = v58;
  v57 = 0x800000000LL;
  v11 = sub_395B1B0((_QWORD *)*v48, a2, &v52, a6 + 8, (int *)a6, 8u);
  v13 = (unsigned int)v57;
  if ( (unsigned int)v57 >= HIDWORD(v57) )
  {
    v44 = v11;
    sub_16CD150((__int64)&v56, v58, 0, 8, v11, v12);
    v13 = (unsigned int)v57;
    v11 = v44;
  }
  v56[v13] = v11;
  LODWORD(v57) = v57 + 1;
  v42 = *(_DWORD *)a6 == 2;
  v14 = sub_1644900(v9, 1u);
  v15 = sub_159C470(v14, v42, 0);
  v17 = (unsigned int)v57;
  if ( (unsigned int)v57 >= HIDWORD(v57) )
  {
    v45 = v15;
    sub_16CD150((__int64)&v56, v58, 0, 8, v15, v16);
    v17 = (unsigned int)v57;
    v15 = v45;
  }
  v56[v17] = v15;
  LODWORD(v57) = v57 + 1;
  v20 = sub_395B1B0((_QWORD *)*v48, a2, &v52, a6 + 56, (int *)(a6 + 4), 8u);
  v21 = (unsigned int)v57;
  if ( (unsigned int)v57 >= HIDWORD(v57) )
  {
    sub_16CD150((__int64)&v56, v58, 0, 8, v18, v19);
    v21 = (unsigned int)v57;
  }
  v56[v21] = v20;
  LODWORD(v57) = v57 + 1;
  v22 = *(_DWORD *)(a6 + 4) == 2;
  v23 = sub_1644900(v9, 1u);
  v26 = sub_159C470(v23, v22, 0);
  v27 = (unsigned int)v57;
  if ( (unsigned int)v57 >= HIDWORD(v57) )
  {
    sub_16CD150((__int64)&v56, v58, 0, 8, v24, v25);
    v27 = (unsigned int)v57;
  }
  v56[v27] = v26;
  LODWORD(v57) = v57 + 1;
  v30 = sub_39590E0(v9, a6, a7, &v52);
  v31 = (unsigned int)v57;
  if ( (unsigned int)v57 >= HIDWORD(v57) )
  {
    sub_16CD150((__int64)&v56, v58, 0, 8, v28, v29);
    v31 = (unsigned int)v57;
  }
  v56[v31] = v30;
  LODWORD(v57) = v57 + 1;
  v32 = sub_15E26F0(v48, 4005, 0, 0);
  v33 = (unsigned int)v57;
  v34 = v32;
  v55 = 1;
  v53 = "idp4a";
  v54 = 3;
  v49 = v56;
  v41 = v57 + 1;
  v43 = *(_QWORD *)(*(_QWORD *)v32 + 24LL);
  v35 = sub_1648AB0(72, (int)v57 + 1, 0);
  v36 = (__int64)v35;
  if ( v35 )
  {
    sub_15F1EA0((__int64)v35, **(_QWORD **)(v43 + 16), 54, (__int64)&v35[-3 * v33 - 3], v41, 0);
    *(_QWORD *)(v36 + 56) = 0;
    sub_15F5B40(v36, v43, v34, v49, v33, (__int64)&v53, 0, 0);
  }
  sub_15F2180(v36, v52);
  v37 = *(_DWORD *)(a6 + 112);
  *(_QWORD *)a1 = v36;
  v38 = *(_QWORD *)a6 == 0x200000002LL;
  *(_QWORD *)(a1 + 16) = v36;
  v52 = v36;
  *(_DWORD *)(a1 + 12) = v37;
  *(_DWORD *)(a1 + 8) = v38 + 1;
  if ( v37 != *(_DWORD *)(a7 + 12) )
  {
    sub_395A2F0((__int64)&v50, a3, a4, (_DWORD *)a1, (_DWORD *)a7);
    v39 = _mm_loadu_si128(&v50);
    *(_QWORD *)(a1 + 16) = v51;
    *(__m128i *)a1 = v39;
  }
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  return a1;
}
