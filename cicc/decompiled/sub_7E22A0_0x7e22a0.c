// Function: sub_7E22A0
// Address: 0x7e22a0
//
__m128i *__fastcall sub_7E22A0(char *a1, unsigned __int8 a2)
{
  char *v2; // r12
  _QWORD *v3; // rax
  __m128i *result; // rax

  v2 = (char *)sub_815620(a1);
  v3 = sub_72BA30(a2);
  result = sub_7E2190(v2, 1, (__int64)v3, 0);
  result[5].m128i_i8[9] = result[5].m128i_i8[9] & 0xD7 | (32 * (unk_4D042C8 & 1)) | 8;
  return result;
}
