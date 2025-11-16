// Function: sub_2D1FAE0
// Address: 0x2d1fae0
//
void __fastcall sub_2D1FAE0(__int64 a1)
{
  __int64 *v1; // rax
  struct __jmp_buf_tag *v2; // r12
  __int64 *v3; // r13
  _BYTE *v4; // rax

  if ( *(_BYTE *)(a1 + 8) )
  {
    v1 = sub_CEACC0();
    v2 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v1);
    if ( v2 )
    {
      v3 = sub_CEAD60();
      v4 = (_BYTE *)sub_CEECD0(1, 1u);
      *v4 = 1;
      sub_C94E10((__int64)v3, v4);
      longjmp(v2, 1);
    }
  }
}
