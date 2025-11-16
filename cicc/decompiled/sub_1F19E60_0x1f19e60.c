// Function: sub_1F19E60
// Address: 0x1f19e60
//
unsigned __int64 __fastcall sub_1F19E60(
        _QWORD *a1,
        int a2,
        __int32 a3,
        int a4,
        __int64 a5,
        unsigned __int64 *a6,
        char a7,
        int a8)
{
  __int64 v8; // rbx
  int v10; // r8d
  __int64 v11; // r13
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 result; // rax
  __int64 v16; // r13
  int v17; // r9d
  unsigned __int64 v18; // rdx
  unsigned int v19; // r14d
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  unsigned int v24; // eax
  unsigned __int64 v25; // rbx
  __int64 v26; // r13
  int v27; // eax
  __int64 v28; // r12
  __int64 v29; // r10
  __int64 v30; // rdi
  int v31; // r8d
  int v32; // r9d
  unsigned int v33; // r10d
  __int64 v34; // rax
  unsigned int v35; // r15d
  __int64 (__fastcall *v36)(__int64, __int64); // rax
  int v37; // r12d
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  int i; // r15d
  unsigned int *v41; // r14
  int v42; // ebx
  signed int v43; // ebx
  unsigned int v44; // r12d
  unsigned int v45; // r13d
  unsigned int v46; // r14d
  __int64 v47; // rcx
  __int64 v48; // rdi
  _QWORD *v49; // rsi
  _QWORD *v50; // rdx
  unsigned __int64 v51; // [rsp+10h] [rbp-C0h]
  unsigned int v52; // [rsp+10h] [rbp-C0h]
  __int64 v53; // [rsp+18h] [rbp-B8h]
  unsigned int v59; // [rsp+40h] [rbp-90h]
  unsigned int v60; // [rsp+40h] [rbp-90h]
  unsigned int v61; // [rsp+44h] [rbp-8Ch]
  int v62; // [rsp+48h] [rbp-88h]
  __int64 v63; // [rsp+48h] [rbp-88h]
  __int64 v64; // [rsp+48h] [rbp-88h]
  __int64 v65; // [rsp+50h] [rbp-80h]
  int v66; // [rsp+58h] [rbp-78h]
  __int64 v67; // [rsp+58h] [rbp-78h]
  signed int v68; // [rsp+58h] [rbp-78h]
  int v69; // [rsp+58h] [rbp-78h]
  __int64 v70; // [rsp+68h] [rbp-68h] BYREF
  __m128i v71; // [rsp+70h] [rbp-60h] BYREF
  __int64 v72; // [rsp+80h] [rbp-50h] BYREF
  __int64 v73; // [rsp+88h] [rbp-48h]
  __int64 v74; // [rsp+90h] [rbp-40h]

  v8 = *(_QWORD *)(a1[6] + 8LL);
  if ( a4 == -1 || (unsigned int)sub_1E69F40(a1[4], a2) == a4 )
  {
    v70 = 0;
    v11 = *(_QWORD *)(a5 + 56);
    v12 = (unsigned __int64)sub_1E0B640(v11, v8 + 960, &v70, 0);
    sub_1DD5BA0((__int64 *)(a5 + 16), v12);
    v13 = *a6;
    *(_QWORD *)(v12 + 8) = a6;
    *(_QWORD *)v12 = v13 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v12 & 7LL;
    *(_QWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v12;
    v14 = *a6;
    v71.m128i_i32[1] = 0;
    v72 = 0;
    *a6 = v12 | v14 & 7;
    v73 = 0;
    v71.m128i_i32[2] = a3;
    v74 = 0;
    v71.m128i_i32[0] = 0x10000000;
    sub_1E1A9C0(v12, v11, &v71);
    v71.m128i_i64[0] = 0;
    v72 = 0;
    v71.m128i_i32[2] = a2;
    v73 = 0;
    v74 = 0;
    sub_1E1A9C0(v12, v11, &v71);
    if ( v70 )
      sub_161E7C0((__int64)&v70, v70);
    return sub_1DC1550(*(_QWORD *)(a1[2] + 272LL), v12, a7) & 0xFFFFFFFFFFFFFFF8LL | 4;
  }
  v16 = a1[2];
  v17 = *(_DWORD *)(**(_QWORD **)(a1[9] + 16LL) + 4LL * (unsigned int)(*(_DWORD *)(a1[9] + 64LL) + a8));
  v18 = *(unsigned int *)(v16 + 408);
  v19 = v17 & 0x7FFFFFFF;
  v20 = v17 & 0x7FFFFFFF;
  v21 = 8 * v20;
  if ( (v17 & 0x7FFFFFFFu) < (unsigned int)v18 )
  {
    v53 = *(_QWORD *)(*(_QWORD *)(v16 + 400) + 8LL * v19);
    if ( v53 )
      goto LABEL_9;
  }
  v46 = v19 + 1;
  if ( (unsigned int)v18 < v46 )
  {
    if ( v46 < v18 )
    {
      *(_DWORD *)(v16 + 408) = v46;
    }
    else if ( v46 > v18 )
    {
      if ( v46 > (unsigned __int64)*(unsigned int *)(v16 + 412) )
      {
        v69 = *(_DWORD *)(**(_QWORD **)(a1[9] + 16LL) + 4LL * (unsigned int)(*(_DWORD *)(a1[9] + 64LL) + a8));
        v64 = 8LL * (v17 & 0x7FFFFFFF);
        sub_16CD150(v16 + 400, (const void *)(v16 + 416), v46, 8, v10, v17);
        v18 = *(unsigned int *)(v16 + 408);
        v21 = v64;
        v17 = v69;
      }
      v47 = *(_QWORD *)(v16 + 400);
      v48 = *(_QWORD *)(v16 + 416);
      v49 = (_QWORD *)(v47 + 8LL * v46);
      v50 = (_QWORD *)(v47 + 8 * v18);
      if ( v49 != v50 )
      {
        do
          *v50++ = v48;
        while ( v49 != v50 );
        v47 = *(_QWORD *)(v16 + 400);
      }
      *(_DWORD *)(v16 + 408) = v46;
      goto LABEL_37;
    }
  }
  v47 = *(_QWORD *)(v16 + 400);
LABEL_37:
  *(_QWORD *)(v47 + v21) = sub_1DBA290(v17);
  v53 = *(_QWORD *)(*(_QWORD *)(v16 + 400) + 8 * v20);
  sub_1DBB110((_QWORD *)v16, v53);
LABEL_9:
  v71.m128i_i64[0] = (__int64)&v72;
  v22 = a1[4];
  v23 = a1[7];
  v71.m128i_i64[1] = 0x800000000LL;
  v24 = *(_DWORD *)(v23 + 104);
  v25 = *(_QWORD *)(*(_QWORD *)(v22 + 24) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 <= 1 )
    goto LABEL_46;
  v26 = v24 - 1;
  v66 = a4;
  v27 = ~a4;
  v28 = 0;
  v61 = 0;
  v59 = 0;
  v62 = v27;
  while ( 1 )
  {
    v35 = v28 + 1;
    v36 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 112LL);
    if ( v36 != sub_1E15B90 )
    {
      if ( v25 != ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v36)(v23, v25, v35) )
      {
        if ( v26 == ++v28 )
          goto LABEL_22;
        goto LABEL_18;
      }
      v23 = a1[7];
    }
    ++v28;
    v29 = 4 * v28;
    v30 = *(unsigned int *)(*(_QWORD *)(v23 + 248) + 4 * v28);
    if ( (_DWORD)v30 == v66 )
      break;
    if ( ((unsigned int)v30 & v62) == 0 )
    {
      v33 = sub_39FAC40(v30);
      v34 = v71.m128i_u32[2];
      if ( v71.m128i_i32[2] >= (unsigned __int32)v71.m128i_i32[3] )
      {
        v52 = v33;
        sub_16CD150((__int64)&v71, &v72, 0, 4, v31, v32);
        v34 = v71.m128i_u32[2];
        v33 = v52;
      }
      *(_DWORD *)(v71.m128i_i64[0] + 4 * v34) = v35;
      ++v71.m128i_i32[2];
      if ( v61 < v33 )
      {
        v61 = v33;
        v59 = v35;
      }
    }
    if ( v26 == v28 )
    {
LABEL_22:
      v37 = v66;
      if ( v59 )
      {
        v29 = 4LL * v59;
        goto LABEL_24;
      }
LABEL_46:
      sub_16BD130("Impossible to implement partial COPY", 1u);
    }
LABEL_18:
    v23 = a1[7];
  }
  v59 = v35;
  v37 = v66;
LABEL_24:
  v67 = v29;
  v38 = sub_1F19C20(a1, a2, a3, a5, a6, v59, v53, a7, 0);
  v39 = a1[7];
  v51 = v38;
  for ( i = v37 & ~*(_DWORD *)(*(_QWORD *)(v39 + 248) + v67); i; i &= ~*(_DWORD *)(*(_QWORD *)(v39 + 248) + 4LL * v45) )
  {
    v63 = v71.m128i_i64[0] + 4LL * v71.m128i_u32[2];
    if ( v71.m128i_i64[0] == v63 )
      goto LABEL_46;
    v68 = 0x80000000;
    v41 = (unsigned int *)v71.m128i_i64[0];
    v60 = 0;
    v65 = *(_QWORD *)(v39 + 248);
    while ( 1 )
    {
      v44 = *(_DWORD *)(v65 + 4LL * *v41);
      v45 = *v41;
      if ( i == v44 )
        break;
      v42 = sub_39FAC40(v44 & i);
      v43 = v42 - sub_39FAC40(~i & v44);
      if ( v43 > v68 )
      {
        v68 = v43;
        v60 = v45;
      }
      if ( (unsigned int *)v63 == ++v41 )
      {
        v45 = v60;
        break;
      }
    }
    if ( !v45 )
      goto LABEL_46;
    sub_1F19C20(a1, a2, a3, a5, a6, v45, v53, a7, v51);
    v39 = a1[7];
  }
  result = v51;
  if ( (__int64 *)v71.m128i_i64[0] != &v72 )
  {
    _libc_free(v71.m128i_u64[0]);
    return v51;
  }
  return result;
}
