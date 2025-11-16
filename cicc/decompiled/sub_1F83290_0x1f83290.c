// Function: sub_1F83290
// Address: 0x1f83290
//
__int64 *__fastcall sub_1F83290(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int128 a10)
{
  __int64 v10; // r11
  unsigned __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rax
  __int64 *result; // rax
  __int64 v16; // rax
  __int128 *v17; // rcx
  const void ***v18; // r10
  int v19; // r8d
  __int64 *v20; // r9
  unsigned __int8 *v21; // rax
  const void **v22; // r15
  unsigned int v23; // r14d
  __int128 v24; // rax
  __int64 v25; // r9
  const void ***v26; // r8
  int v27; // r15d
  bool v28; // al
  char v29; // al
  __int64 v30; // r9
  __int128 *v31; // r12
  __int64 *v32; // r13
  __int64 v33; // rsi
  const void ***v34; // r15
  int v35; // r14d
  __int128 v36; // [rsp-30h] [rbp-D0h]
  __int128 v37; // [rsp-10h] [rbp-B0h]
  const void ***v38; // [rsp+8h] [rbp-98h]
  int v39; // [rsp+10h] [rbp-90h]
  __int128 *v40; // [rsp+18h] [rbp-88h]
  const void ***v41; // [rsp+18h] [rbp-88h]
  __int64 v42; // [rsp+20h] [rbp-80h]
  __int64 *v43; // [rsp+20h] [rbp-80h]
  __int128 v44; // [rsp+20h] [rbp-80h]
  unsigned __int64 v46; // [rsp+20h] [rbp-80h]
  __int64 *v47; // [rsp+30h] [rbp-70h]
  __int64 *v48; // [rsp+30h] [rbp-70h]
  __int64 *v49; // [rsp+30h] [rbp-70h]
  __int64 v50; // [rsp+40h] [rbp-60h]
  __int64 *v51; // [rsp+40h] [rbp-60h]
  __int64 *v52; // [rsp+40h] [rbp-60h]
  __int64 v54; // [rsp+40h] [rbp-60h]
  __int64 v56; // [rsp+50h] [rbp-50h] BYREF
  int v57; // [rsp+58h] [rbp-48h]
  __int64 v58; // [rsp+60h] [rbp-40h] BYREF
  int v59; // [rsp+68h] [rbp-38h]

  v10 = a4;
  v12 = a5;
  v13 = *(unsigned __int16 *)(a2 + 24);
  if ( v13 != 52 && ((_DWORD)a3 || v13 != 71) )
    goto LABEL_4;
  v28 = sub_1D185B0(a4);
  v12 = a5;
  v10 = a4;
  if ( !v28 )
    goto LABEL_4;
  v46 = a5;
  v54 = v10;
  v29 = sub_1D18C40(a6, 1);
  v10 = v54;
  v12 = v46;
  if ( !v29 )
  {
    v31 = *(__int128 **)(a2 + 32);
    v32 = *(__int64 **)a1;
    v33 = *(_QWORD *)(a6 + 72);
    v34 = *(const void ****)(a6 + 40);
    v35 = *(_DWORD *)(a6 + 60);
    v58 = v33;
    if ( v33 )
      sub_1623A60((__int64)&v58, v33, 2);
    v59 = *(_DWORD *)(a6 + 64);
    result = sub_1D37470(v32, 68, (__int64)&v58, v34, v35, v30, *v31, *(__int128 *)((char *)v31 + 40), a10);
    if ( v58 )
    {
      v49 = result;
      sub_161E7C0((__int64)&v58, v58);
      return v49;
    }
  }
  else
  {
LABEL_4:
    v14 = sub_1F6DE40(*(_DWORD **)(a1 + 8), v10, v12);
    if ( v14
      && *(_WORD *)(v14 + 24) == 71
      && DWORD2(a10) == 1
      && *(_WORD *)(a10 + 24) == 68
      && (v50 = v14, sub_1D185B0(*(_QWORD *)(*(_QWORD *)(a10 + 32) + 40LL)))
      && (v16 = *(_QWORD *)(a10 + 32), v50 == *(_QWORD *)v16)
      && !*(_DWORD *)(v16 + 8) )
    {
      v17 = *(__int128 **)(v50 + 32);
      v18 = *(const void ****)(v50 + 40);
      v19 = *(_DWORD *)(v50 + 60);
      v20 = *(__int64 **)a1;
      v58 = *(_QWORD *)(a6 + 72);
      if ( v58 )
      {
        v38 = v18;
        v39 = v19;
        v40 = v17;
        v42 = v16;
        v51 = v20;
        sub_1F6CA20(&v58);
        v18 = v38;
        v19 = v39;
        v17 = v40;
        v16 = v42;
        v20 = v51;
      }
      v59 = *(_DWORD *)(a6 + 64);
      v43 = sub_1D37470(
              v20,
              68,
              (__int64)&v58,
              v18,
              v19,
              (__int64)v20,
              *v17,
              *(__int128 *)((char *)v17 + 40),
              *(_OWORD *)(v16 + 80));
      sub_17CD270(&v58);
      sub_1F81BC0(a1, (__int64)v43);
      v47 = v43;
      v52 = *(__int64 **)a1;
      v21 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
      v22 = (const void **)*((_QWORD *)v21 + 1);
      v23 = *v21;
      v58 = *(_QWORD *)(a6 + 72);
      if ( v58 )
        sub_1F6CA20(&v58);
      v59 = *(_DWORD *)(a6 + 64);
      *(_QWORD *)&v24 = sub_1D38BB0((__int64)v52, 0, (__int64)&v58, v23, v22, 0, a7, a8, a9, 0);
      v26 = *(const void ****)(a6 + 40);
      v27 = *(_DWORD *)(a6 + 60);
      v56 = *(_QWORD *)(a6 + 72);
      if ( v56 )
      {
        v41 = v26;
        v44 = v24;
        sub_1F6CA20(&v56);
        v26 = v41;
        v24 = v44;
      }
      *((_QWORD *)&v37 + 1) = 1;
      *(_QWORD *)&v37 = v47;
      *((_QWORD *)&v36 + 1) = a3;
      *(_QWORD *)&v36 = a2;
      v57 = *(_DWORD *)(a6 + 64);
      v48 = sub_1D37470(v52, 68, (__int64)&v56, v26, v27, v25, v36, v24, v37);
      sub_17CD270(&v56);
      sub_17CD270(&v58);
      return v48;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
