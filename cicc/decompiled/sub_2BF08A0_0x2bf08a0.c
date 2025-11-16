// Function: sub_2BF08A0
// Address: 0x2bf08a0
//
void __fastcall sub_2BF08A0(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  _BYTE *v4; // [rsp-30h] [rbp-30h] BYREF

  if ( a3 )
  {
    if ( *a2 > 0x1Cu )
    {
      v4 = a3;
      sub_9B8FE0((__int64)a2, (__int64 *)&v4, 1);
      sub_2BF0870(a1, (__int64)a2, a3);
    }
  }
}
