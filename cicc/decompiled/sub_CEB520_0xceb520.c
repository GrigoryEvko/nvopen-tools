// Function: sub_CEB520
// Address: 0xceb520
//
struct __jmp_buf_tag *__fastcall sub_CEB520(_QWORD *a1, __int64 a2, __int64 a3, char *a4)
{
  __int64 *v4; // rax
  struct __jmp_buf_tag *result; // rax
  struct __jmp_buf_tag *v6; // r12
  __int64 *v7; // r13
  _BYTE *v8; // rax

  sub_CEB020(a1, 0x100u, 1, a4);
  v4 = sub_CEACC0();
  result = (struct __jmp_buf_tag *)sub_C94E20((__int64)v4);
  if ( result )
  {
    v6 = result;
    v7 = sub_CEAD60();
    v8 = (_BYTE *)sub_CEECD0(1, 1);
    *v8 = 1;
    sub_C94E10((__int64)v7, v8);
    longjmp(v6, 1);
  }
  return result;
}
