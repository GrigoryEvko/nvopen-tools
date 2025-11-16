// Function: sub_A5BC40
// Address: 0xa5bc40
//
bool __fastcall sub_A5BC40(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbp
  bool result; // al
  _QWORD v6[5]; // [rsp-28h] [rbp-28h] BYREF

  if ( (a1[7] & 0x10) != 0 || *a1 <= 3u || (result = *a1 > 0x15u && *a1 != 24) )
  {
    v6[4] = v4;
    v6[2] = a3;
    v6[0] = off_4979428;
    v6[1] = 0;
    v6[3] = a4;
    sub_A5A730(a2, (__int64)a1, (__int64)v6);
    return 1;
  }
  return result;
}
