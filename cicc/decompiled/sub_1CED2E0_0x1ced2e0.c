// Function: sub_1CED2E0
// Address: 0x1ced2e0
//
__int64 __fastcall sub_1CED2E0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  struct __jmp_buf_tag *v4; // r12
  __int64 *v5; // r13
  _BYTE *v6; // rax

  if ( sub_1CECFA0(a1, a2) )
  {
    v3 = sub_1C3E710();
    v4 = (struct __jmp_buf_tag *)sub_16D40F0((__int64)v3);
    if ( v4 )
    {
      v5 = sub_1C3E7B0();
      v6 = (_BYTE *)sub_1C42D70(1, 1);
      *v6 = 1;
      sub_16D40E0((__int64)v5, v6);
      longjmp(v4, 1);
    }
  }
  return 0;
}
