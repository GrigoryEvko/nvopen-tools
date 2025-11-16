// Function: sub_1D490E0
// Address: 0x1d490e0
//
void (*__fastcall sub_1D490E0(__int64 a1))()
{
  void (*v1)(void); // rax
  __int64 v2; // rsi
  const __m128i *v3; // rax
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  _QWORD *v15; // rax
  unsigned __int64 v16; // rdx
  const __m128i *v17; // r14
  _QWORD *v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // r15
  void (*result)(); // rax
  __m128i v22; // [rsp+0h] [rbp-120h] BYREF
  __int64 v23; // [rsp+10h] [rbp-110h]
  __int64 v24; // [rsp+18h] [rbp-108h]
  __int64 v25; // [rsp+20h] [rbp-100h]
  __int64 v26; // [rsp+28h] [rbp-F8h]
  __m128i v27; // [rsp+30h] [rbp-F0h]
  _QWORD *v28; // [rsp+48h] [rbp-D8h] BYREF
  __int64 (__fastcall **v29)(); // [rsp+50h] [rbp-D0h] BYREF
  __int64 v30; // [rsp+58h] [rbp-C8h]
  __int64 v31; // [rsp+60h] [rbp-C0h]
  _QWORD *v32; // [rsp+68h] [rbp-B8h]
  _QWORD v33[7]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+A8h] [rbp-78h]
  int v35; // [rsp+B0h] [rbp-70h]
  __int64 v36; // [rsp+B8h] [rbp-68h]
  int v37; // [rsp+C0h] [rbp-60h]
  __int64 v38; // [rsp+C8h] [rbp-58h] BYREF
  __int64 v39; // [rsp+D0h] [rbp-50h]
  _QWORD *v40; // [rsp+D8h] [rbp-48h]
  __int64 v41; // [rsp+E0h] [rbp-40h]
  __int64 v42; // [rsp+E8h] [rbp-38h] BYREF

  v1 = *(void (**)(void))(*(_QWORD *)a1 + 192LL);
  if ( v1 != nullsub_688 )
    v1();
  v2 = 0;
  *(_DWORD *)(a1 + 416) = sub_1D236A0(*(_QWORD *)(a1 + 272));
  v3 = *(const __m128i **)(a1 + 272);
  v4 = v3[11].m128i_i64[0];
  v22 = _mm_loadu_si128(v3 + 11);
  v9 = sub_1D274F0(1u, v5, v6, v7, v8);
  v35 = 0;
  v10 = _mm_load_si128(&v22);
  v33[5] = v9;
  v34 = 0x100000000LL;
  v27 = v10;
  v42 = 0;
  v33[6] = 0;
  v41 = 0;
  v36 = 0;
  v37 = -65536;
  v40 = v33;
  LODWORD(v39) = v10.m128i_i32[2];
  v38 = v10.m128i_i64[0];
  v11 = *(_QWORD *)(v4 + 48);
  memset(v33, 0, 24);
  v33[3] = -4294967084LL;
  v42 = v11;
  if ( v11 )
    *(_QWORD *)(v11 + 24) = &v42;
  v41 = v4 + 48;
  *(_QWORD *)(v4 + 48) = &v38;
  v12 = *(_QWORD *)(a1 + 272);
  v33[4] = &v38;
  v13 = *(_QWORD *)(v12 + 176);
  LODWORD(v34) = 1;
  if ( !v13 )
    BUG();
  v14 = *(_QWORD *)(v12 + 664);
  v15 = *(_QWORD **)(v13 + 16);
  v31 = v12;
  v29 = off_49F9C90;
  v30 = v14;
  *(_QWORD *)(v12 + 664) = &v29;
  v28 = v15;
  v32 = &v28;
LABEL_12:
  v17 = *(const __m128i **)(a1 + 272);
  v18 = (_QWORD *)v17[12].m128i_i64[1];
  while ( v15 != v18 )
  {
    v16 = *v15 & 0xFFFFFFFFFFFFFFF8LL;
    v28 = (_QWORD *)v16;
    v15 = (_QWORD *)v16;
    if ( !v16 )
      BUG();
    if ( *(_QWORD *)(v16 + 40) )
    {
      v2 = v16 - 8;
      if ( (unsigned int)(*(__int16 *)(v16 + 16) - 81) <= 0x11 )
        v2 = sub_1D44F30(v17, v2);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 208LL))(a1, v2);
      v15 = v28;
      goto LABEL_12;
    }
  }
  v19 = v38;
  v20 = v39;
  if ( v38 )
  {
    nullsub_686();
    v26 = v20;
    v2 = 0;
    v25 = v19;
    v17[11].m128i_i64[0] = v19;
    v17[11].m128i_i32[2] = v26;
    sub_1D23870();
  }
  else
  {
    v24 = v39;
    v23 = 0;
    v17[11].m128i_i64[0] = 0;
    v17[11].m128i_i32[2] = v24;
  }
  *(_QWORD *)(v31 + 664) = v30;
  sub_1D189A0((__int64)v33);
  result = *(void (**)())(*(_QWORD *)a1 + 200LL);
  if ( result != nullsub_689 )
    return (void (*)())((__int64 (__fastcall *)(__int64, __int64))result)(a1, v2);
  return result;
}
