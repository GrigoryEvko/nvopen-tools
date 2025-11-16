// Function: sub_373B2C0
// Address: 0x373b2c0
//
__int64 __fastcall sub_373B2C0(__int64 a1, __int64 a2)
{
  __int64 (*v4)(); // rax
  unsigned int v5; // r13d
  char v6; // r15
  __int64 v7; // rdi
  __int64 *v8; // r14
  __m128i v9; // rax
  unsigned __int8 v10; // al
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __m128i v21; // xmm1
  __int64 v22; // r10
  __m128i v23; // xmm0
  __int64 result; // rax
  __int64 v25; // rax
  __m128i v26; // xmm2
  __int64 v27; // [rsp+30h] [rbp-120h]
  unsigned int v28; // [rsp+40h] [rbp-110h] BYREF
  char v29; // [rsp+48h] [rbp-108h]
  unsigned int v30; // [rsp+50h] [rbp-100h] BYREF
  char v31; // [rsp+58h] [rbp-F8h]
  __m128i v32; // [rsp+60h] [rbp-F0h] BYREF
  char v33; // [rsp+70h] [rbp-E0h]
  __m128i v34; // [rsp+80h] [rbp-D0h] BYREF
  char v35; // [rsp+90h] [rbp-C0h]
  __m128i v36; // [rsp+A0h] [rbp-B0h]
  char v37; // [rsp+B0h] [rbp-A0h]
  __m128i v38; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v39; // [rsp+D0h] [rbp-80h]
  __int64 v40; // [rsp+E0h] [rbp-70h]
  __int64 v41; // [rsp+E8h] [rbp-68h]
  __int64 v42; // [rsp+F0h] [rbp-60h]
  __m128i v43; // [rsp+100h] [rbp-50h]
  __int64 v44; // [rsp+110h] [rbp-40h]

  v4 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 184) + 224LL) + 104LL);
  if ( v4 == sub_C13EF0 || !(unsigned __int8)v4() )
  {
    v5 = *(_DWORD *)(a1 + 72);
    if ( !a2 )
      goto LABEL_21;
LABEL_3:
    if ( *(_QWORD *)(a1 + 744) == a2 )
      return *(unsigned int *)(a1 + 752);
    *(_QWORD *)(a1 + 744) = a2;
    v6 = 0;
    v7 = *(_QWORD *)(a2 + 40);
    v8 = *(__int64 **)(*(_QWORD *)(a1 + 184) + 224LL);
    if ( v7 )
    {
      v9.m128i_i64[0] = sub_B91420(v7);
      v6 = 1;
      v38 = v9;
    }
    sub_3222AF0(&v32, *(_QWORD *)(a1 + 208), a2);
    v10 = *(_BYTE *)(a2 - 16);
    v11 = a2 - 16;
    if ( (v10 & 2) != 0 )
    {
      v12 = **(_QWORD **)(a2 - 32);
      if ( !v12 )
      {
        v15 = 0;
        goto LABEL_16;
      }
    }
    else
    {
      v12 = *(_QWORD *)(v11 - 8LL * ((v10 >> 2) & 0xF));
      if ( !v12 )
      {
        v15 = 0;
        goto LABEL_9;
      }
    }
    v13 = sub_B91420(v12);
    v11 = a2 - 16;
    v12 = v13;
    v10 = *(_BYTE *)(a2 - 16);
    v15 = v14;
    if ( (v10 & 2) == 0 )
    {
LABEL_9:
      v16 = v11 - 8LL * ((v10 >> 2) & 0xF);
      goto LABEL_10;
    }
LABEL_16:
    v16 = *(_QWORD *)(a2 - 32);
LABEL_10:
    v17 = *(_QWORD *)(v16 + 8);
    if ( v17 )
    {
      v27 = v12;
      v18 = sub_B91420(v17);
      v12 = v27;
      v17 = v18;
      v20 = v19;
    }
    else
    {
      v20 = 0;
    }
    LOBYTE(v39) = v6;
    v21 = _mm_loadu_si128(&v38);
    v22 = *v8;
    v44 = v39;
    v23 = _mm_loadu_si128(&v32);
    v43 = v21;
    v37 = v33;
    v36 = v23;
    (*(void (__fastcall **)(unsigned int *, __int64 *, _QWORD, __int64, __int64, _QWORD, __int64, __int64, __int64, __int64, char, __int64, __int64, __int64))(v22 + 656))(
      &v30,
      v8,
      0,
      v17,
      v20,
      v5,
      v12,
      v15,
      v23.m128i_i64[0],
      v23.m128i_i64[1],
      v33,
      v21.m128i_i64[0],
      v21.m128i_i64[1],
      v39);
    if ( (v31 & 1) == 0 )
    {
      result = v30;
      *(_DWORD *)(a1 + 752) = v30;
      return result;
    }
LABEL_24:
    BUG();
  }
  v5 = 0;
  if ( a2 )
    goto LABEL_3;
LABEL_21:
  v25 = *(_QWORD *)(a1 + 184);
  LOBYTE(v42) = 0;
  v26 = _mm_loadu_si128(&v34);
  v35 = 0;
  (*(void (__fastcall **)(unsigned int *, _QWORD, _QWORD, const char *, _QWORD, _QWORD, const char *, _QWORD, __int64, __int64, _BYTE, __int64, __int64, __int64))(**(_QWORD **)(v25 + 224) + 656LL))(
    &v28,
    *(_QWORD *)(v25 + 224),
    0,
    byte_3F871B3,
    0,
    v5,
    byte_3F871B3,
    0,
    v26.m128i_i64[0],
    v26.m128i_i64[1],
    0,
    v40,
    v41,
    v42);
  if ( (v29 & 1) != 0 )
    goto LABEL_24;
  return v28;
}
