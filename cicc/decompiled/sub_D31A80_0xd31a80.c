// Function: sub_D31A80
// Address: 0xd31a80
//
__int64 __fastcall sub_D31A80(__int64 a1, unsigned __int8 *a2, _BYTE *a3, int a4)
{
  unsigned __int8 *v5; // rax
  __int64 v6; // r8
  unsigned __int8 *v7; // r15
  unsigned __int16 v8; // ax
  __int64 v9; // r13
  bool v10; // al
  unsigned __int64 v11; // r14
  int v12; // r13d
  _QWORD *v13; // rbx
  __int64 v14; // r14
  int v15; // r12d
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int8 **v21; // rdi
  unsigned __int8 **v22; // rbx
  unsigned __int8 **v23; // r14
  _QWORD *v24; // rdi
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  unsigned __int8 *v29; // [rsp+10h] [rbp-110h]
  unsigned __int8 v30; // [rsp+1Fh] [rbp-101h]
  __int64 v31; // [rsp+20h] [rbp-100h]
  _BYTE *v33; // [rsp+30h] [rbp-F0h]
  _QWORD *v34; // [rsp+38h] [rbp-E8h]
  __int64 v35; // [rsp+38h] [rbp-E8h]
  __int64 v36; // [rsp+38h] [rbp-E8h]
  __m128i v37; // [rsp+40h] [rbp-E0h] BYREF
  __m128i v38; // [rsp+50h] [rbp-D0h] BYREF
  __m128i v39; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v40[3]; // [rsp+70h] [rbp-B0h] BYREF
  char v41; // [rsp+A0h] [rbp-80h]
  unsigned __int8 **v42; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v43; // [rsp+B8h] [rbp-68h]
  _BYTE v44[96]; // [rsp+C0h] [rbp-60h] BYREF

  v29 = a2;
  v33 = (_BYTE *)sub_B43CC0(a1);
  v5 = sub_BD3990(*(unsigned __int8 **)(a1 - 32), (__int64)a2);
  v6 = 0;
  v7 = v5;
  v8 = *(_WORD *)(a1 + 2);
  if ( ((v8 >> 7) & 6) != 0 )
    return v6;
  if ( (v8 & 1) != 0 )
    return v6;
  v9 = *(_QWORD *)(a1 + 40);
  v31 = *(_QWORD *)(a1 + 8);
  v10 = sub_B46500((unsigned __int8 *)a1);
  v6 = 0;
  v30 = v10;
  v11 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  v42 = (unsigned __int8 **)v44;
  v43 = 0x600000000LL;
  v34 = (_QWORD *)(v9 + 48);
  if ( v11 == v9 + 48 )
    return v6;
  v12 = a4;
  v13 = (_QWORD *)v11;
  while ( 1 )
  {
    v14 = 0;
    if ( v13 )
      v14 = (__int64)(v13 - 3);
    if ( sub_B46AA0(v14) )
      goto LABEL_5;
    v15 = v12 - 1;
    if ( !v12 )
      goto LABEL_19;
    a2 = v7;
    v16 = sub_D301A0(v14, v7, v31, v30, v33, a3);
    if ( v16 )
      break;
    --v12;
    if ( (unsigned __int8)sub_B46490(v14) )
    {
      v19 = (unsigned int)v43;
      v20 = (unsigned int)v43 + 1LL;
      if ( v20 > HIDWORD(v43) )
      {
        a2 = v44;
        sub_C8D5F0((__int64)&v42, v44, v20, 8u, v17, v18);
        v19 = (unsigned int)v43;
      }
      v12 = v15;
      v42[v19] = (unsigned __int8 *)v14;
      LODWORD(v43) = v43 + 1;
    }
LABEL_5:
    v13 = (_QWORD *)(*v13 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v34 == v13 )
      goto LABEL_19;
  }
  v35 = v16;
  a2 = (unsigned __int8 *)a1;
  sub_D665A0(&v37);
  v21 = v42;
  v6 = v35;
  v22 = &v42[(unsigned int)v43];
  if ( v22 == v42 )
    goto LABEL_20;
  v23 = v42;
  while ( 1 )
  {
    a2 = *v23;
    v24 = *(_QWORD **)v29;
    v25 = _mm_loadu_si128(&v37);
    v26 = _mm_loadu_si128(&v38);
    v41 = 1;
    v27 = _mm_loadu_si128(&v39);
    v40[0] = v25;
    v40[1] = v26;
    v40[2] = v27;
    if ( (sub_CF63E0(v24, a2, v40, (__int64)(v29 + 8)) & 2) != 0 )
      break;
    if ( v22 == ++v23 )
    {
      v6 = v35;
      v21 = v42;
      goto LABEL_20;
    }
  }
LABEL_19:
  v21 = v42;
  v6 = 0;
LABEL_20:
  if ( v21 != (unsigned __int8 **)v44 )
  {
    v36 = v6;
    _libc_free(v21, a2);
    return v36;
  }
  return v6;
}
