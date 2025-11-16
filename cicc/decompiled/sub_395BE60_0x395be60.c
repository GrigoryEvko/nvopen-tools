// Function: sub_395BE60
// Address: 0x395be60
//
__int64 __fastcall sub_395BE60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rdx
  _QWORD *v11; // r13
  __int64 v12; // rax
  __int64 v13; // r8
  int v14; // r9d
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r8
  int v18; // r9d
  __int64 v19; // rax
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // r14
  __int64 v23; // rax
  _BOOL8 v24; // r14
  __int64 v25; // rax
  int v26; // r8d
  int v27; // r9d
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // r14
  __int64 v34; // rax
  int v35; // r8d
  int v36; // r9d
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r15
  __int64 v41; // r13
  _QWORD *v42; // rax
  __int64 v43; // r14
  int v44; // ecx
  bool v45; // zf
  __m128i v46; // xmm0
  int v50; // [rsp+18h] [rbp-118h]
  _BOOL8 v51; // [rsp+20h] [rbp-110h]
  __int64 v52; // [rsp+20h] [rbp-110h]
  __int64 v53; // [rsp+20h] [rbp-110h]
  __int64 v54; // [rsp+20h] [rbp-110h]
  __int64 *v55; // [rsp+38h] [rbp-F8h]
  __int64 *v56; // [rsp+38h] [rbp-F8h]
  __m128i v57; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+50h] [rbp-E0h]
  __int64 v59; // [rsp+68h] [rbp-C8h] BYREF
  const char *v60; // [rsp+70h] [rbp-C0h] BYREF
  char v61; // [rsp+80h] [rbp-B0h]
  char v62; // [rsp+81h] [rbp-AFh]
  _QWORD *v63; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v64; // [rsp+98h] [rbp-98h]
  _QWORD v65[2]; // [rsp+A0h] [rbp-90h] BYREF
  __int64 *v66; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v67; // [rsp+B8h] [rbp-78h]
  _BYTE v68[112]; // [rsp+C0h] [rbp-70h] BYREF

  v10 = *(_QWORD *)(a6 + 104);
  v11 = **(_QWORD ***)(a5 + 40);
  v55 = *(__int64 **)(a5 + 40);
  v63 = v65;
  v64 = 0x200000000LL;
  v59 = v10;
  v12 = *(_QWORD *)(a7 + 16);
  if ( v12 )
  {
    v65[0] = v10;
    v65[1] = v12;
    LODWORD(v64) = 2;
    v59 = sub_3958EE0(a3, (__int64 *)&v63, a4);
  }
  v66 = (__int64 *)v68;
  v67 = 0x800000000LL;
  v13 = sub_395B1B0((_QWORD *)*v55, a2, &v59, a6 + 8, (int *)a6, 0x10u);
  v15 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
    v53 = v13;
    sub_16CD150((__int64)&v66, v68, 0, 8, v13, v14);
    v15 = (unsigned int)v67;
    v13 = v53;
  }
  v66[v15] = v13;
  LODWORD(v67) = v67 + 1;
  v51 = *(_DWORD *)a6 == 2;
  v16 = sub_1644900(v11, 1u);
  v17 = sub_159C470(v16, v51, 0);
  v19 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
    v54 = v17;
    sub_16CD150((__int64)&v66, v68, 0, 8, v17, v18);
    v19 = (unsigned int)v67;
    v17 = v54;
  }
  v66[v19] = v17;
  LODWORD(v67) = v67 + 1;
  v22 = sub_395B1B0((_QWORD *)*v55, a2, &v59, a6 + 56, (int *)(a6 + 4), 8u);
  v23 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
    sub_16CD150((__int64)&v66, v68, 0, 8, v20, v21);
    v23 = (unsigned int)v67;
  }
  v66[v23] = v22;
  LODWORD(v67) = v67 + 1;
  v24 = *(_DWORD *)(a6 + 4) == 2;
  v25 = sub_1644900(v11, 1u);
  v28 = sub_159C470(v25, v24, 0);
  v29 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
    sub_16CD150((__int64)&v66, v68, 0, 8, v26, v27);
    v29 = (unsigned int)v67;
  }
  v66[v29] = v28;
  LODWORD(v67) = v67 + 1;
  v30 = sub_1644900(v11, 1u);
  v33 = sub_159C470(v30, 0, 0);
  v34 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
    sub_16CD150((__int64)&v66, v68, 0, 8, v31, v32);
    v34 = (unsigned int)v67;
  }
  v66[v34] = v33;
  LODWORD(v67) = v67 + 1;
  v37 = sub_39590E0(v11, a6, a7, &v59);
  v38 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
    sub_16CD150((__int64)&v66, v68, 0, 8, v35, v36);
    v38 = (unsigned int)v67;
  }
  v66[v38] = v37;
  LODWORD(v67) = v67 + 1;
  v39 = sub_15E26F0(v55, 4004, 0, 0);
  v40 = (unsigned int)v67;
  v41 = v39;
  v62 = 1;
  v60 = "idp2a";
  v61 = 3;
  v56 = v66;
  v50 = v67 + 1;
  v52 = *(_QWORD *)(*(_QWORD *)v39 + 24LL);
  v42 = sub_1648AB0(72, (int)v67 + 1, 0);
  v43 = (__int64)v42;
  if ( v42 )
  {
    sub_15F1EA0((__int64)v42, **(_QWORD **)(v52 + 16), 54, (__int64)&v42[-3 * v40 - 3], v50, 0);
    *(_QWORD *)(v43 + 56) = 0;
    sub_15F5B40(v43, v52, v41, v56, v40, (__int64)&v60, 0, 0);
  }
  sub_15F2180(v43, v59);
  v44 = *(_DWORD *)(a6 + 112);
  *(_QWORD *)a1 = v43;
  v45 = *(_QWORD *)a6 == 0x200000002LL;
  *(_QWORD *)(a1 + 16) = v43;
  v59 = v43;
  *(_DWORD *)(a1 + 12) = v44;
  *(_DWORD *)(a1 + 8) = v45 + 1;
  if ( v44 != *(_DWORD *)(a7 + 12) )
  {
    sub_395A2F0((__int64)&v57, a3, a4, (_DWORD *)a1, (_DWORD *)a7);
    v46 = _mm_loadu_si128(&v57);
    *(_QWORD *)(a1 + 16) = v58;
    *(__m128i *)a1 = v46;
  }
  if ( v66 != (__int64 *)v68 )
    _libc_free((unsigned __int64)v66);
  if ( v63 != v65 )
    _libc_free((unsigned __int64)v63);
  return a1;
}
