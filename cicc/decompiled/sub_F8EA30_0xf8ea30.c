// Function: sub_F8EA30
// Address: 0xf8ea30
//
void __fastcall sub_F8EA30(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v4; // rdx
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 | a2 )
  {
    v5[0] = sub_AA48A0(*(_QWORD *)(a1 + 40));
    v4 = sub_B8C2F0(v5, a2, a3, 0);
  }
  else
  {
    v4 = 0;
  }
  sub_B99FD0(a1, 2u, v4);
}
