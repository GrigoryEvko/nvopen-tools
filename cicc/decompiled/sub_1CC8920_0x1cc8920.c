// Function: sub_1CC8920
// Address: 0x1cc8920
//
__int64 __fastcall sub_1CC8920(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  bool v7; // zf
  _QWORD *v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned __int64 v21; // rbx
  __int64 v22; // rdi
  unsigned __int64 v23; // rax
  _QWORD *v24; // rax
  _QWORD *v25; // r8
  __int64 v26; // rcx
  __int64 v27; // rsi
  __m128i v28; // xmm1
  __m128i v29; // xmm0
  _QWORD *v30; // rdi
  char v31; // r15
  char v32; // al
  _QWORD *v33; // rax
  _QWORD *v34; // r8
  __int64 v35; // rdx
  _QWORD *v36; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v37; // [rsp+10h] [rbp-B0h] BYREF
  unsigned __int64 v38; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v39; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 i; // [rsp+28h] [rbp-98h] BYREF
  __m128i v41; // [rsp+30h] [rbp-90h] BYREF
  __m128i v42; // [rsp+40h] [rbp-80h] BYREF
  __int64 v43; // [rsp+50h] [rbp-70h]
  __m128i v44[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v45; // [rsp+80h] [rbp-40h]
  char v46; // [rsp+88h] [rbp-38h]

  v7 = *(_BYTE *)(a2 + 16) == 54;
  v38 = a2;
  v37 = a3;
  if ( !v7 )
  {
    if ( sub_1C30740(a2) )
    {
      v21 = v38;
      if ( v38 )
        v21 = v38 + 24;
      while ( 1 )
      {
        v23 = v37;
        if ( v37 )
          v23 = v37 + 24;
        if ( v23 == v21 )
          break;
        v22 = v21 - 24;
        if ( !v21 )
          v22 = 0;
        if ( (unsigned __int8)sub_15F3040(v22) )
          return 0;
        v21 = *(_QWORD *)(v21 + 8);
      }
    }
    return 1;
  }
  sub_141EB40(&v41, (__int64 *)a2);
  v36 = a5 + 1;
  v11 = a5 + 1;
  v12 = (_QWORD *)a5[2];
  if ( !v12 )
  {
    v11 = a5 + 1;
LABEL_9:
    v44[0].m128i_i64[0] = (__int64)&v38;
    v11 = (_QWORD *)sub_1CC72C0(a5, v11, (unsigned __int64 **)v44);
    goto LABEL_10;
  }
  do
  {
    while ( 1 )
    {
      v13 = v12[2];
      v14 = v12[3];
      if ( v12[4] >= v38 )
        break;
      v12 = (_QWORD *)v12[3];
      if ( !v14 )
        goto LABEL_7;
    }
    v11 = v12;
    v12 = (_QWORD *)v12[2];
  }
  while ( v13 );
LABEL_7:
  if ( v36 == v11 || v11[4] > v38 )
    goto LABEL_9;
LABEL_10:
  v15 = v11[5];
  v39 = v15;
  if ( v15 )
  {
    v16 = (_QWORD *)a6[2];
    v17 = a6 + 1;
    if ( !v16 )
      goto LABEL_35;
    do
    {
      while ( 1 )
      {
        v18 = v16[2];
        v19 = v16[3];
        if ( v16[4] >= v15 )
          break;
        v16 = (_QWORD *)v16[3];
        if ( !v19 )
          goto LABEL_16;
      }
      v17 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v18 );
LABEL_16:
    if ( a6 + 1 == v17 || v17[4] > v15 )
    {
LABEL_35:
      v44[0].m128i_i64[0] = (__int64)&v39;
      v17 = (_QWORD *)sub_1CC72C0(a6, v17, (unsigned __int64 **)v44);
    }
    a4 = v17[5];
    v39 = a4;
  }
  else
  {
    v39 = a4;
  }
  if ( !a4 || !(unsigned __int8)sub_1CC8170(a1 + 112, a4, v37) )
    return 1;
  v24 = (_QWORD *)a5[2];
  if ( v24 )
  {
    v25 = a5 + 1;
    do
    {
      v26 = v24[2];
      if ( v24[4] < v37 )
      {
        v24 = (_QWORD *)v24[3];
      }
      else
      {
        v25 = v24;
        v24 = (_QWORD *)v24[2];
      }
    }
    while ( v24 );
    if ( v36 != v25 && v25[4] <= v37 )
      goto LABEL_46;
  }
  else
  {
    v25 = a5 + 1;
  }
  v44[0].m128i_i64[0] = (__int64)&v37;
  v25 = (_QWORD *)sub_1CC72C0(a5, v25, (unsigned __int64 **)v44);
LABEL_46:
  v27 = v25[5];
  for ( i = v27; ; i = v27 )
  {
    v28 = _mm_loadu_si128(&v42);
    v46 = 1;
    v29 = _mm_loadu_si128(&v41);
    v30 = *(_QWORD **)(a1 + 8);
    v44[1] = v28;
    v44[0] = v29;
    v45 = v43;
    v31 = sub_13575E0(v30, v27, v44, v26);
    v32 = sub_1C30710(i);
    if ( (v31 & 2) != 0 && v32 != 1 )
      break;
    if ( i == v39 )
      return 1;
    v33 = (_QWORD *)a5[2];
    v34 = a5 + 1;
    if ( !v33 )
      goto LABEL_57;
    do
    {
      while ( 1 )
      {
        v26 = v33[2];
        v35 = v33[3];
        if ( v33[4] >= i )
          break;
        v33 = (_QWORD *)v33[3];
        if ( !v35 )
          goto LABEL_55;
      }
      v34 = v33;
      v33 = (_QWORD *)v33[2];
    }
    while ( v26 );
LABEL_55:
    if ( v36 == v34 || v34[4] > i )
    {
LABEL_57:
      v44[0].m128i_i64[0] = (__int64)&i;
      v34 = (_QWORD *)sub_1CC72C0(a5, v34, (unsigned __int64 **)v44);
    }
    v27 = v34[5];
  }
  return 0;
}
