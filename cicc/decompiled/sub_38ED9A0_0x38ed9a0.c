// Function: sub_38ED9A0
// Address: 0x38ed9a0
//
__int64 __fastcall sub_38ED9A0(__int64 a1)
{
  __int64 v2; // rax
  _DWORD *v3; // rax
  unsigned int v4; // r13d
  __int64 v6; // rax
  int v7; // eax
  __int64 v8; // r10
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r11
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdi
  int v16; // edx
  __int64 (*v17)(); // rax
  char v18; // al
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  __m128i *v21; // rax
  int v22; // eax
  __int64 v23; // [rsp+0h] [rbp-100h]
  unsigned __int64 v24; // [rsp+0h] [rbp-100h]
  unsigned __int64 v25; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v26; // [rsp+8h] [rbp-F8h]
  __int64 v27; // [rsp+8h] [rbp-F8h]
  __int64 *v28; // [rsp+10h] [rbp-F0h]
  __int64 v29; // [rsp+10h] [rbp-F0h]
  __int64 v30; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v31; // [rsp+20h] [rbp-E0h]
  __int64 v32; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v33; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v34; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+48h] [rbp-B8h] BYREF
  _QWORD v36[2]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v37; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v38[2]; // [rsp+70h] [rbp-90h] BYREF
  _BYTE v39[16]; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 v40[2]; // [rsp+90h] [rbp-70h] BYREF
  __int64 v41; // [rsp+A0h] [rbp-60h] BYREF
  const char *v42; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v43; // [rsp+B8h] [rbp-48h]
  _OWORD v44[4]; // [rsp+C0h] [rbp-40h] BYREF

  v38[0] = (unsigned __int64)v39;
  v38[1] = 0;
  v39[0] = 0;
  v2 = sub_3909460(a1);
  v32 = sub_39092A0(v2);
  v42 = "expected string in '.incbin' directive";
  LOWORD(v44[0]) = 259;
  v3 = (_DWORD *)sub_3909460(a1);
  if ( (unsigned __int8)sub_3909CB0(a1, *v3 != 3, &v42) || (unsigned __int8)sub_38ECF20(a1, v38) )
    goto LABEL_2;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  if ( !(unsigned __int8)sub_3909EB0(a1, 25) )
    goto LABEL_8;
  if ( *(_DWORD *)sub_3909460(a1) != 25
    && ((unsigned __int8)sub_3909470(a1, &v35) || (unsigned __int8)sub_38EB9C0(a1, &v33)) )
  {
    goto LABEL_2;
  }
  if ( (unsigned __int8)sub_3909EB0(a1, 25) )
  {
    v20 = sub_3909460(a1);
    v31 = sub_39092A0(v20);
    v42 = 0;
    if ( sub_38EB6A0(a1, &v34, (__int64)&v42) )
    {
LABEL_2:
      v4 = 1;
      goto LABEL_3;
    }
  }
  else
  {
LABEL_8:
    v31 = 0;
  }
  LOWORD(v44[0]) = 259;
  v42 = "unexpected token in '.incbin' directive";
  if ( (unsigned __int8)sub_3909E20(a1, 9, &v42) )
    goto LABEL_2;
  v42 = "skip is negative";
  LOWORD(v44[0]) = 259;
  v4 = sub_3909C80(a1, v33 >> 63, v35, &v42);
  if ( (_BYTE)v4 )
    goto LABEL_2;
  v28 = *(__int64 **)(a1 + 344);
  v23 = v34;
  v25 = v33;
  v42 = (const char *)v44;
  v43 = 0;
  LOBYTE(v44[0]) = 0;
  v6 = sub_3909290(a1 + 144);
  v7 = sub_16CF050(v28, v38, v6, &v42);
  if ( !v7 )
  {
    if ( v42 != (const char *)v44 )
      j_j___libc_free_0((unsigned __int64)v42);
LABEL_31:
    sub_8FD6D0((__int64)v40, "Could not find incbin file '", v38);
    if ( v40[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v21 = (__m128i *)sub_2241490(v40, "'", 1u);
    v42 = (const char *)v44;
    if ( (__m128i *)v21->m128i_i64[0] == &v21[1] )
    {
      v44[0] = _mm_loadu_si128(v21 + 1);
    }
    else
    {
      v42 = (const char *)v21->m128i_i64[0];
      *(_QWORD *)&v44[0] = v21[1].m128i_i64[0];
    }
    v43 = v21->m128i_i64[1];
    v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
    v21->m128i_i64[1] = 0;
    v21[1].m128i_i8[0] = 0;
    v37 = 260;
    v36[0] = &v42;
    v4 = sub_3909790(a1, v32, v36, 0, 0);
    if ( v42 != (const char *)v44 )
      j_j___libc_free_0((unsigned __int64)v42);
    v19 = v40[0];
    if ( (__int64 *)v40[0] != &v41 )
    {
LABEL_22:
      j_j___libc_free_0(v19);
      goto LABEL_3;
    }
    goto LABEL_3;
  }
  v8 = v23;
  v9 = 0;
  v10 = *(_QWORD *)(**(_QWORD **)(a1 + 344) + 24LL * (unsigned int)(v7 - 1));
  v11 = *(_QWORD *)(v10 + 16);
  v12 = *(_QWORD *)(v10 + 8);
  v13 = v11 - v12;
  if ( v11 - v12 >= v25 )
  {
    v11 = v12 + v25;
    v9 = v13 - v25;
  }
  v14 = v9;
  if ( !v23 )
  {
LABEL_21:
    (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64))(**(_QWORD **)(a1 + 328) + 400LL))(
      *(_QWORD *)(a1 + 328),
      v11,
      v14);
    v19 = (unsigned __int64)v42;
    if ( v42 == (const char *)v44 )
      goto LABEL_3;
    goto LABEL_22;
  }
  v15 = *(_QWORD *)(a1 + 328);
  v16 = 0;
  v17 = *(__int64 (**)())(*(_QWORD *)v15 + 72LL);
  if ( v17 != sub_168DB40 )
  {
    v24 = v9;
    v27 = v11;
    v30 = v8;
    v22 = ((__int64 (__fastcall *)(__int64, unsigned __int64 *, _QWORD))v17)(v15, v38, 0);
    v9 = v24;
    v11 = v27;
    v8 = v30;
    v16 = v22;
  }
  v26 = v9;
  v29 = v11;
  v18 = sub_38CF2B0(v8, v36, v16);
  v11 = v29;
  if ( v18 )
  {
    v14 = v36[0];
    if ( v36[0] >= 0LL )
    {
      if ( v36[0] > v26 )
        v14 = v26;
      goto LABEL_21;
    }
    v40[0] = (unsigned __int64)"negative count has no effect";
    LOWORD(v41) = 259;
    v4 = sub_38E4170((_QWORD *)a1, v31, (__int64)v40, 0, 0);
  }
  else
  {
    v40[0] = (unsigned __int64)"expected absolute expression";
    LOWORD(v41) = 259;
    v4 = sub_3909790(a1, v31, v40, 0, 0);
  }
  if ( v42 != (const char *)v44 )
    j_j___libc_free_0((unsigned __int64)v42);
  if ( (_BYTE)v4 )
    goto LABEL_31;
LABEL_3:
  if ( (_BYTE *)v38[0] != v39 )
    j_j___libc_free_0(v38[0]);
  return v4;
}
