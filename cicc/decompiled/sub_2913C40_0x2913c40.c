// Function: sub_2913C40
// Address: 0x2913c40
//
__int64 *__fastcall sub_2913C40(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  char *v20; // rdi
  __int64 v22; // [rsp+0h] [rbp-1B0h] BYREF
  __int64 v23; // [rsp+8h] [rbp-1A8h]
  __int64 v24; // [rsp+10h] [rbp-1A0h]
  __int64 v25; // [rsp+18h] [rbp-198h]
  char *v26; // [rsp+20h] [rbp-190h] BYREF
  __int64 v27; // [rsp+28h] [rbp-188h]
  _BYTE v28[32]; // [rsp+30h] [rbp-180h] BYREF
  __int64 v29; // [rsp+50h] [rbp-160h]
  __int64 v30; // [rsp+58h] [rbp-158h]
  __int64 v31; // [rsp+60h] [rbp-150h]
  __int64 v32; // [rsp+68h] [rbp-148h]
  __int64 v33; // [rsp+70h] [rbp-140h]
  __int64 v34; // [rsp+78h] [rbp-138h]
  char *v35; // [rsp+80h] [rbp-130h] BYREF
  __int64 v36; // [rsp+88h] [rbp-128h]
  _BYTE v37[32]; // [rsp+90h] [rbp-120h] BYREF
  __int64 v38; // [rsp+B0h] [rbp-100h]
  __int64 v39; // [rsp+B8h] [rbp-F8h]
  __int64 v40; // [rsp+C0h] [rbp-F0h]
  __int64 v41; // [rsp+C8h] [rbp-E8h]
  __int64 v42; // [rsp+D0h] [rbp-E0h]
  __int64 v43; // [rsp+D8h] [rbp-D8h]
  char *v44; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v45; // [rsp+E8h] [rbp-C8h]
  _BYTE v46[32]; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+110h] [rbp-A0h]
  __int64 v48; // [rsp+118h] [rbp-98h]
  __int64 v49; // [rsp+120h] [rbp-90h]
  __int64 v50; // [rsp+128h] [rbp-88h]
  __int64 v51; // [rsp+130h] [rbp-80h]
  __int64 v52; // [rsp+138h] [rbp-78h]
  char *v53; // [rsp+140h] [rbp-70h] BYREF
  __int64 v54; // [rsp+148h] [rbp-68h]
  _BYTE v55[32]; // [rsp+150h] [rbp-60h] BYREF
  __int64 v56; // [rsp+170h] [rbp-40h]
  __int64 v57; // [rsp+178h] [rbp-38h]

  v7 = *(unsigned int *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 24);
  v31 = 0;
  v32 = 0;
  v35 = v37;
  v36 = 0x400000000LL;
  v33 = v8 + 24 * v7;
  v34 = v33;
  v38 = v33;
  v39 = 0;
  v22 = 0;
  v23 = 0;
  v24 = v8;
  v25 = v8;
  v26 = v28;
  v27 = 0x400000000LL;
  v29 = v33;
  v30 = 0;
  if ( v8 == v33 )
  {
    v51 = v8;
    v49 = 0;
    v50 = 0;
    v52 = v8;
    v53 = v55;
    v54 = 0x400000000LL;
  }
  else
  {
    sub_2912870((__int64)&v22, a2, v8, a4, a5, a6);
    v8 = (unsigned int)v36;
    v53 = v55;
    v54 = 0x400000000LL;
    v49 = v31;
    v50 = v32;
    v51 = v33;
    v52 = v34;
    if ( (_DWORD)v36 )
      sub_2913AE0((__int64)&v53, &v35, (unsigned int)v36, v9, v10, v11);
  }
  v12 = v22;
  v13 = v23;
  v14 = v24;
  v44 = v46;
  v56 = v38;
  v15 = v25;
  v16 = (unsigned int)v27;
  v40 = v22;
  v57 = v39;
  v41 = v23;
  v42 = v24;
  v43 = v25;
  v45 = 0x400000000LL;
  if ( (_DWORD)v27 )
  {
    sub_2913AE0((__int64)&v44, &v26, v8, (unsigned int)v27, v23, v22);
    v12 = v40;
    v13 = v41;
    v14 = v42;
    v15 = v43;
    v16 = (unsigned int)v45;
  }
  a1[3] = v15;
  v17 = v29;
  v18 = v30;
  a1[4] = (__int64)(a1 + 6);
  v47 = v17;
  v48 = v18;
  *a1 = v12;
  a1[1] = v13;
  a1[2] = v14;
  a1[5] = 0x400000000LL;
  if ( (_DWORD)v16 )
  {
    sub_2913AE0((__int64)(a1 + 4), &v44, v17, v16, v13, v12);
    v17 = v47;
    v18 = v48;
  }
  a1[11] = v18;
  v19 = v49;
  a1[10] = v17;
  a1[12] = v19;
  a1[13] = v50;
  a1[14] = v51;
  a1[15] = v52;
  a1[16] = (__int64)(a1 + 18);
  a1[17] = 0x400000000LL;
  if ( (_DWORD)v54 )
    sub_2913AE0((__int64)(a1 + 16), &v53, v17, v16, v13, v12);
  v20 = v44;
  a1[22] = v56;
  a1[23] = v57;
  if ( v20 != v46 )
    _libc_free((unsigned __int64)v20);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return a1;
}
