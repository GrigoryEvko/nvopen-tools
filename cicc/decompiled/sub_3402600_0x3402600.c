// Function: sub_3402600
// Address: 0x3402600
//
unsigned __int8 *__fastcall sub_3402600(
        _QWORD *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  __int64 v7; // r15
  __int64 v8; // r13
  unsigned __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // r8
  __int16 v14; // ax
  __int16 v15; // r12
  unsigned int v16; // eax
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 *v20; // rdx
  unsigned __int8 *result; // rax
  __int64 v22; // rdx
  __int64 v23; // r8
  __int16 v24; // ax
  int v25; // r9d
  bool v26; // al
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int128 v29; // [rsp-10h] [rbp-1A0h]
  __int64 v30; // [rsp+8h] [rbp-188h]
  unsigned int v31; // [rsp+8h] [rbp-188h]
  __int64 v32; // [rsp+10h] [rbp-180h]
  unsigned __int8 *v33; // [rsp+10h] [rbp-180h]
  unsigned __int8 *v34; // [rsp+10h] [rbp-180h]
  __int64 v35; // [rsp+18h] [rbp-178h]
  __int64 v36; // [rsp+20h] [rbp-170h] BYREF
  __int64 v37; // [rsp+28h] [rbp-168h]
  unsigned __int64 v38; // [rsp+30h] [rbp-160h] BYREF
  unsigned int v39; // [rsp+38h] [rbp-158h]
  unsigned __int64 v40; // [rsp+40h] [rbp-150h] BYREF
  unsigned int v41; // [rsp+48h] [rbp-148h]
  _BYTE *v42; // [rsp+50h] [rbp-140h] BYREF
  __int64 v43; // [rsp+58h] [rbp-138h]
  _BYTE v44[304]; // [rsp+60h] [rbp-130h] BYREF

  v7 = (__int64)a2;
  v8 = a5;
  v36 = a3;
  v37 = a4;
  if ( !(_WORD)a3 )
  {
    v31 = a3;
    v26 = sub_3007100((__int64)&v36);
    LOWORD(a3) = v31;
    if ( !v26 )
      goto LABEL_3;
    v24 = sub_3009970((__int64)&v36, (__int64)a2, v31, v27, a5);
    v23 = v28;
LABEL_32:
    sub_34007B0((__int64)a1, v8, (__int64)a2, v24, v23, 1u, a7, 0);
    return sub_33FAF80((__int64)a1, 170, (__int64)a2, (unsigned int)v36, v37, v25, a7);
  }
  if ( (unsigned __int16)(a3 - 176) <= 0x34u )
  {
    v23 = 0;
    v24 = word_4456580[(unsigned __int16)a3 - 1];
    goto LABEL_32;
  }
LABEL_3:
  v9 = 0;
  v42 = v44;
  v43 = 0x1000000000LL;
  while ( 1 )
  {
    if ( (_WORD)a3 )
    {
      if ( (unsigned __int16)(a3 - 176) > 0x34u )
        goto LABEL_5;
    }
    else if ( !sub_3007100((__int64)&v36) )
    {
      break;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v36 )
      break;
    if ( (unsigned __int16)(v36 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_5:
    v10 = (unsigned __int16)v36;
    v11 = (unsigned __int16)v36 - 1;
    v12 = word_4456340[v11];
    if ( v12 <= v9 )
      goto LABEL_22;
    if ( (_WORD)v36 )
    {
      v13 = 0;
      v14 = word_4456580[v11];
      goto LABEL_8;
    }
LABEL_27:
    v14 = sub_3009970((__int64)&v36, (__int64)a2, v10, v12, a5);
    v13 = v22;
LABEL_8:
    v15 = v14;
    v39 = *(_DWORD *)(v8 + 8);
    if ( v39 > 0x40 )
    {
      v30 = v13;
      sub_C43780((__int64)&v38, (const void **)v8);
      v13 = v30;
    }
    else
    {
      v38 = *(_QWORD *)v8;
    }
    v32 = v13;
    sub_C47170((__int64)&v38, v9);
    v16 = v39;
    a2 = &v40;
    v39 = 0;
    v41 = v16;
    v40 = v38;
    v17 = sub_34007B0((__int64)a1, (__int64)&v40, v7, v15, v32, 0, a7, 0);
    a6 = v18;
    v19 = (unsigned int)v43;
    a5 = (__int64)v17;
    if ( (unsigned __int64)(unsigned int)v43 + 1 > HIDWORD(v43) )
    {
      a2 = (unsigned __int64 *)v44;
      v34 = v17;
      v35 = a6;
      sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 0x10u, (__int64)v17, a6);
      v19 = (unsigned int)v43;
      a5 = (__int64)v34;
      a6 = v35;
    }
    v20 = (__int64 *)&v42[16 * v19];
    *v20 = a5;
    v20[1] = a6;
    LODWORD(v43) = v43 + 1;
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    if ( v39 > 0x40 && v38 )
      j_j___libc_free_0_0(v38);
    LOWORD(a3) = v36;
    ++v9;
  }
  if ( v9 < (unsigned int)sub_3007130((__int64)&v36, (__int64)a2) )
    goto LABEL_27;
LABEL_22:
  *((_QWORD *)&v29 + 1) = (unsigned int)v43;
  *(_QWORD *)&v29 = v42;
  result = sub_33FC220(a1, 156, v7, v36, v37, a6, v29);
  if ( v42 != v44 )
  {
    v33 = result;
    _libc_free((unsigned __int64)v42);
    return v33;
  }
  return result;
}
