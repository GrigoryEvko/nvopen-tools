// Function: sub_1213C20
// Address: 0x1213c20
//
__int64 __fastcall sub_1213C20(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v6; // rax
  __m128i *v7; // rsi
  __int32 v8; // [rsp+Ch] [rbp-34h] BYREF
  __int64 v9; // [rsp+10h] [rbp-30h] BYREF
  __int64 v10[5]; // [rsp+18h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 440, "expected 'callee' here") )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
    return 1;
  v6 = *(_QWORD *)(a1 + 232);
  v9 = 0;
  v10[0] = v6;
  if ( (unsigned __int8)sub_12122D0(a1, &v9, &v8) )
    return 1;
  a2[1] = v9;
  v7 = *(__m128i **)(a3 + 8);
  if ( v7 == *(__m128i **)(a3 + 16) )
  {
    sub_1213A90((const __m128i **)a3, v7, &v8, v10);
  }
  else
  {
    if ( v7 )
    {
      v7->m128i_i32[0] = v8;
      v7->m128i_i64[1] = v10[0];
      v7 = *(__m128i **)(a3 + 8);
    }
    *(_QWORD *)(a3 + 8) = v7 + 1;
  }
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_1211570(a1, a2)
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_12115D0(a1, (__int64)(a2 + 2)) )
  {
    return 1;
  }
  else
  {
    return sub_120AFE0(a1, 13, "expected ')' here");
  }
}
