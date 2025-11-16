// Function: sub_3022230
// Address: 0x3022230
//
__int64 __fastcall sub_3022230(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  const char *v4; // rax
  __int8 *v5; // r15
  size_t v6; // rax
  size_t v7; // r8
  __m128i *v8; // rdx
  size_t v9; // rax
  __int64 v11; // rax
  __m128i *v12; // rdi
  size_t n; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+18h] [rbp-58h] BYREF
  __m128i *v15; // [rsp+20h] [rbp-50h] BYREF
  size_t v16; // [rsp+28h] [rbp-48h]
  __m128i v17[4]; // [rsp+30h] [rbp-40h] BYREF

  v3 = sub_3936750();
  if ( !(unsigned __int8)sub_314C600(a2, v3) )
  {
    sub_39367A0(v3);
    *(_QWORD *)(a1 + 32) = 0;
    *(_OWORD *)a1 = 0;
    *(_OWORD *)(a1 + 16) = 0;
    return a1;
  }
  v4 = (const char *)sub_3936860(v3, 0);
  v15 = v17;
  v5 = (__int8 *)v4;
  if ( !v4 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v6 = strlen(v4);
  v14 = v6;
  v7 = v6;
  if ( v6 > 0xF )
  {
    n = v6;
    v11 = sub_22409D0((__int64)&v15, (unsigned __int64 *)&v14, 0);
    v7 = n;
    v15 = (__m128i *)v11;
    v12 = (__m128i *)v11;
    v17[0].m128i_i64[0] = v14;
    goto LABEL_14;
  }
  if ( v6 != 1 )
  {
    if ( !v6 )
    {
      v8 = v17;
      goto LABEL_6;
    }
    v12 = v17;
LABEL_14:
    memcpy(v12, v5, v7);
    v6 = v14;
    v8 = v15;
    goto LABEL_6;
  }
  v17[0].m128i_i8[0] = *v5;
  v8 = v17;
LABEL_6:
  v16 = v6;
  v8->m128i_i8[v6] = 0;
  sub_39367A0(v3);
  *(_QWORD *)a1 = a1 + 16;
  if ( v15 == v17 )
  {
    *(__m128i *)(a1 + 16) = _mm_load_si128(v17);
  }
  else
  {
    *(_QWORD *)a1 = v15;
    *(_QWORD *)(a1 + 16) = v17[0].m128i_i64[0];
  }
  v9 = v16;
  *(_BYTE *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 8) = v9;
  return a1;
}
