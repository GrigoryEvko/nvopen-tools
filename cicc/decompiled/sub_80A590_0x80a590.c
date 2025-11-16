// Function: sub_80A590
// Address: 0x80a590
//
__int64 __fastcall sub_80A590(_BYTE *a1)
{
  __int64 v1; // rbp
  char v2; // al
  int v4; // [rsp-18h] [rbp-18h] BYREF
  __int64 v5; // [rsp-8h] [rbp-8h]

  v2 = a1[140];
  if ( (unsigned __int8)(v2 - 9) > 2u && (v2 != 2 || (a1[161] & 8) == 0) || (a1[143] & 0xC0) == 0 )
    return 0;
  v5 = v1;
  sub_737670((__int64)a1, 6u, (__int64 (__fastcall *)(__int64, _QWORD, _DWORD *))sub_809F00, &v4, 15);
  return 0;
}
