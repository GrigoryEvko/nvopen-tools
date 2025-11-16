// Function: sub_2A8A3C0
// Address: 0x2a8a3c0
//
char __fastcall sub_2A8A3C0(__int64 *a1, unsigned int a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v7; // [rsp+8h] [rbp-48h]

  LOBYTE(v4) = 1;
  if ( (1LL << a4) % (unsigned __int64)a2 )
  {
    v7 = a1[5];
    sub_B2BE50(*a1);
    LOBYTE(v4) = sub_DFAE90(v7);
    if ( (_BYTE)v4 )
    {
      v5 = a1[5];
      sub_B2BE50(*a1);
      sub_DFAE90(v5);
      return 1;
    }
  }
  return v4;
}
