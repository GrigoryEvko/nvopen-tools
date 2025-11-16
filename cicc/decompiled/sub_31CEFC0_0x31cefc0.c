// Function: sub_31CEFC0
// Address: 0x31cefc0
//
__int64 __fastcall sub_31CEFC0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  struct __jmp_buf_tag *v4; // r12
  __int64 *v5; // r13
  _BYTE *v6; // rax

  if ( sub_31CEC90(a1, a2) )
  {
    v3 = sub_CEACC0();
    v4 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v3);
    if ( v4 )
    {
      v5 = sub_CEAD60();
      v6 = (_BYTE *)sub_CEECD0(1, 1u);
      *v6 = 1;
      sub_C94E10((__int64)v5, v6);
      longjmp(v4, 1);
    }
  }
  return 0;
}
