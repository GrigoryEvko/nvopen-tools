// Function: sub_1CEBDE0
// Address: 0x1cebde0
//
void __fastcall sub_1CEBDE0(__int64 a1)
{
  __int64 *v1; // rax
  struct __jmp_buf_tag *v2; // r12
  __int64 *v3; // r13
  _BYTE *v4; // rax

  if ( *(_BYTE *)(a1 + 8) )
  {
    v1 = sub_1C3E710();
    v2 = (struct __jmp_buf_tag *)sub_16D40F0((__int64)v1);
    if ( v2 )
    {
      v3 = sub_1C3E7B0();
      v4 = (_BYTE *)sub_1C42D70(1, 1);
      *v4 = 1;
      sub_16D40E0((__int64)v3, v4);
      longjmp(v2, 1);
    }
  }
}
