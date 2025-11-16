// Function: sub_39CC330
// Address: 0x39cc330
//
__int64 __fastcall sub_39CC330(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 (*v3)(); // rax
  __int64 v4; // r14
  char v5; // r13
  __m128i *v6; // rax
  __int64 v7; // rdx
  __m128i *v8; // r12
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  void (__fastcall *v16)(unsigned int *, __int64, _QWORD, __int64, __int64, __m128i *, __int64, __int64, __m128i *, __int64); // r11
  __m128i v18; // rax
  void (__fastcall *v19)(__m128i *, __int64, _QWORD, const char *, _QWORD, _QWORD, const char *, _QWORD, __m128i *, __int64); // r11
  char v20; // al
  __int64 v21; // [rsp+0h] [rbp-90h]
  __int64 v22; // [rsp+8h] [rbp-88h]
  unsigned int v23[4]; // [rsp+10h] [rbp-80h] BYREF
  __m128i v24[2]; // [rsp+20h] [rbp-70h] BYREF
  __m128i v25; // [rsp+40h] [rbp-50h] BYREF
  char v26; // [rsp+50h] [rbp-40h]

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 256LL);
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 88LL);
  if ( v3 == sub_168DB60
    || (v20 = ((__int64 (__fastcall *)(_QWORD))v3)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 256LL)),
        v2 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 256LL),
        !v20) )
  {
    v4 = *(unsigned int *)(a1 + 600);
  }
  else
  {
    v4 = 0;
  }
  if ( a2 )
  {
    v5 = *(_BYTE *)(a2 + 56);
    if ( v5 )
    {
      v18.m128i_i64[0] = sub_161E970(*(_QWORD *)(a2 + 48));
      v24[0] = v18;
    }
    v6 = sub_39A3100(a1, a2);
    v7 = *(unsigned int *)(a2 + 8);
    v8 = v6;
    v9 = *(_QWORD *)(a2 - 8 * v7);
    if ( v9 )
    {
      v10 = sub_161E970(*(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
      v22 = v11;
      v7 = *(unsigned int *)(a2 + 8);
      v9 = v10;
    }
    else
    {
      v22 = 0;
    }
    v12 = *(_QWORD *)(a2 + 8 * (1 - v7));
    if ( v12 )
    {
      v21 = v9;
      v13 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v7)));
      v9 = v21;
      v12 = v13;
      v15 = v14;
    }
    else
    {
      v15 = 0;
    }
    v16 = *(void (__fastcall **)(unsigned int *, __int64, _QWORD, __int64, __int64, __m128i *, __int64, __int64, __m128i *, __int64))(*(_QWORD *)v2 + 568LL);
    v26 = v5;
    if ( v5 )
      v25 = _mm_loadu_si128(v24);
    v16(v23, v2, 0, v12, v15, v8, v9, v22, &v25, v4);
    return v23[0];
  }
  else
  {
    v19 = *(void (__fastcall **)(__m128i *, __int64, _QWORD, const char *, _QWORD, _QWORD, const char *, _QWORD, __m128i *, __int64))(*(_QWORD *)v2 + 568LL);
    v26 = 0;
    v19(v24, v2, 0, byte_3F871B3, 0, 0, byte_3F871B3, 0, &v25, v4);
    return v24[0].m128i_u32[0];
  }
}
