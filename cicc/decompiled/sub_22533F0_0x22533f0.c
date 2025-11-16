// Function: sub_22533F0
// Address: 0x22533f0
//
void __fastcall sub_22533F0(unsigned int a1, __int64 a2)
{
  void (__fastcall *v2)(__int64); // rax

  if ( a1 > 1 )
    sub_2207500(*(void (**)(void))(a2 - 56));
  if ( !_InterlockedSub((volatile signed __int32 *)(a2 - 96), 1u) )
  {
    v2 = *(void (__fastcall **)(__int64))(a2 - 72);
    if ( v2 )
      v2(a2 + 32);
    sub_22527D0(a2 + 32);
  }
}
