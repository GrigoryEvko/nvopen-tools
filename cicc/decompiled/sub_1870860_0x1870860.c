// Function: sub_1870860
// Address: 0x1870860
//
char __fastcall sub_1870860(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rax
  unsigned __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_15E4F10(a2);
  v6[0] = v4;
  if ( v4 )
  {
    LOBYTE(v4) = sub_1870060(a1, a2);
    if ( (_BYTE)v4 )
      LOBYTE(v4) = (unsigned __int8)sub_1870740(a3, v6);
  }
  return v4;
}
