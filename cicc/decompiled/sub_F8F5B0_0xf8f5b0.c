// Function: sub_F8F5B0
// Address: 0xf8f5b0
//
char __fastcall sub_F8F5B0(__int64 a1, __int16 a2, _QWORD *a3)
{
  char result; // al
  unsigned __int64 v5; // rdi
  unsigned int v6; // ebx
  unsigned __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 && sub_B91C10(a1, 15) )
    return 1;
  if ( !(unsigned __int8)sub_BC8C50(a1, &v7, v8) )
    return 1;
  v5 = v7;
  if ( !(v8[0] + v7) )
    return 1;
  result = HIBYTE(a2);
  if ( HIBYTE(a2) )
  {
    if ( !(_BYTE)a2 )
      v5 = v8[0];
    v6 = sub_F02DD0(v5, v8[0] + v7);
    return (unsigned int)sub_DF95A0(a3) > v6;
  }
  return result;
}
