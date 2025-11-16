// Function: sub_B86130
// Address: 0xb86130
//
void __fastcall __noreturn sub_B86130(__int64 a1)
{
  __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    sub_B85E70(v2);
    sub_B80B80(*(_QWORD *)(a1 + 8));
  }
  BUG();
}
