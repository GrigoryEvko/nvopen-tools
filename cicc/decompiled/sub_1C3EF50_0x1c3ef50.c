// Function: sub_1C3EF50
// Address: 0x1c3ef50
//
struct __jmp_buf_tag *__fastcall sub_1C3EF50(__int64 a1)
{
  __int64 *v1; // rax
  struct __jmp_buf_tag *result; // rax
  struct __jmp_buf_tag *v3; // r12
  __int64 *v4; // r13
  _BYTE *v5; // rax
  __int16 v6[9]; // [rsp+Eh] [rbp-12h] BYREF

  v6[0] = 256;
  sub_1C3EA60(a1, (char *)v6, 1);
  v1 = sub_1C3E710();
  result = (struct __jmp_buf_tag *)sub_16D40F0((__int64)v1);
  if ( result )
  {
    v3 = result;
    v4 = sub_1C3E7B0();
    v5 = (_BYTE *)sub_1C42D70(1, 1);
    *v5 = 1;
    sub_16D40E0((__int64)v4, v5);
    longjmp(v3, 1);
  }
  return result;
}
