// Function: sub_7EC460
// Address: 0x7ec460
//
__int64 *__fastcall sub_7EC460(__int64 a1, const __m128i *a2)
{
  __m128i *v2; // r13
  __int64 v3; // rax
  _QWORD *v4; // rax
  __int64 *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  void *v11; // rax
  __int64 v12; // rax

  v2 = (__m128i *)sub_724D50(1);
  if ( (unsigned int)sub_7E1F90(a1) || (unsigned int)sub_7E1F40(a1) )
  {
    v3 = sub_7E1E20(a1);
    sub_72BB40(v3, v2);
    sub_7EAFC0(v2);
    v2[8].m128i_i64[0] = a1;
  }
  else
  {
    v12 = sub_72C570();
    sub_72BB40(v12, v2);
    sub_7EB190((__int64)v2, v2);
  }
  v4 = sub_7EBB70((__int64)v2);
  v4[1] = a2->m128i_i64[1];
  v5 = (__int64 *)sub_73E130(v4, a1);
  if ( !(unsigned int)sub_731770((__int64)a2, 0, v6, v7, v8, v9) )
    return v5;
  v11 = sub_730FF0(a2);
  return (__int64 *)sub_73DF90((__int64)v11, v5);
}
