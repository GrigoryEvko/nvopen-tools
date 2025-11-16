// Function: sub_1B423A0
// Address: 0x1b423a0
//
void __fastcall sub_1B423A0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v4; // rdx
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 | a2 )
  {
    v5[0] = sub_157E9C0(*(_QWORD *)(a1 + 40));
    v4 = sub_161BE60(v5, a2, a3);
  }
  else
  {
    v4 = 0;
  }
  sub_1625C10(a1, 2, v4);
}
