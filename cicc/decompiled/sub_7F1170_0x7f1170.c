// Function: sub_7F1170
// Address: 0x7f1170
//
void __fastcall sub_7F1170(const __m128i *a1)
{
  __m128i *v2; // rbx
  __int64 v3; // r13
  const __m128i *v4; // rax
  __int64 *v5; // rbx
  _QWORD *v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdi
  void *v14; // rax

  v2 = (__m128i *)a1[4].m128i_i64[1];
  v3 = a1->m128i_i64[0];
  if ( (unsigned int)sub_7E1E50(v2->m128i_i64[0]) )
  {
    v4 = (const __m128i *)sub_7EC460(v3, v2);
    sub_730620((__int64)a1, v4);
  }
  else
  {
    v5 = sub_7E1E70(v2->m128i_i64);
    if ( dword_4F077C4 == 2 )
      v6 = (_QWORD *)a1->m128i_i64[0];
    else
      v6 = sub_72BA30(5u);
    sub_73D8E0((__int64)a1, 0x3Bu, (__int64)v6, 0, (__int64)v5);
    sub_7F07E0((__int64)a1, 59, v7, v8, v9, v10);
    v13 = a1->m128i_i64[0];
    if ( a1->m128i_i64[0] != v3 && !(unsigned int)sub_8D97D0(v13, v3, 0, v11, v12) )
    {
      v14 = sub_730FF0(a1);
      sub_7E2300((__int64)a1, (__int64)v14, v3);
    }
  }
}
