// Function: sub_2C741A0
// Address: 0x2c741a0
//
__int64 __fastcall sub_2C741A0(__int64 a1)
{
  char v1; // dl
  char v2; // dl
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v6; // [rsp+0h] [rbp-20h]

  v6 = sub_CE8FC0(a1);
  if ( !v1 )
  {
    v6 = sub_CE8F50(a1);
    if ( !v2 )
      return 0;
  }
  v4 = sub_CE90E0(a1);
  v5 = 0;
  if ( BYTE4(v4) )
    v5 = (unsigned int)v4;
  if ( v6 <= 0x400 )
    return (v5 << 32) | (32 * (((v6 - (v6 != 0)) >> 5) + (v6 != 0)));
  else
    return 0;
}
