// Function: sub_DC3A60
// Address: 0xdc3a60
//
__int64 __fastcall sub_DC3A60(__int64 a1, __int64 a2, _BYTE *a3, _BYTE *a4)
{
  _BYTE *v5; // [rsp+8h] [rbp-28h] BYREF
  _BYTE *v6; // [rsp+10h] [rbp-20h] BYREF
  __int64 v7; // [rsp+18h] [rbp-18h] BYREF

  v7 = a2;
  v6 = a3;
  v5 = a4;
  sub_DC7F30(a1, &v7, &v6, &v5, 0);
  if ( (unsigned __int8)sub_DC3800(a1, v7, v6, v5) || (unsigned __int8)sub_DC3B80(a1, v7, v6, v5) )
    return 1;
  else
    return sub_DCD020(a1, v7, v6, v5);
}
