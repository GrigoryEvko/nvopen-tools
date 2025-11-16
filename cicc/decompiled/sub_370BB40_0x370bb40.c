// Function: sub_370BB40
// Address: 0x370bb40
//
void __fastcall sub_370BB40(_QWORD *a1, const __m128i *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  _OWORD v5[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v6; // [rsp+30h] [rbp-20h]

  v3 = a1[7];
  if ( v3 && !a1[5] && !a1[6] && (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v3 + 40LL))(v3) )
  {
    v4 = a2[2].m128i_i64[0];
    v5[0] = _mm_loadu_si128(a2);
    v6 = v4;
    v5[1] = _mm_loadu_si128(a2 + 1);
    if ( (unsigned __int8)v4 > 1u )
      (*(void (__fastcall **)(_QWORD, _OWORD *))(*(_QWORD *)a1[7] + 24LL))(a1[7], v5);
  }
}
