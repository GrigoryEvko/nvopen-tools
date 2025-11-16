// Function: sub_97F890
// Address: 0x97f890
//
bool __fastcall sub_97F890(__int64 a1, _BYTE *a2, size_t a3)
{
  _BYTE *v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  size_t v7; // rdx
  void *s2; // [rsp+0h] [rbp-20h] BYREF
  __int64 v9; // [rsp+8h] [rbp-18h]

  v3 = sub_97E150(a2, a3);
  v9 = v4;
  s2 = v3;
  if ( !v4 )
    return 0;
  v6 = sub_97F810((__int64 *)(a1 + 176), &s2, (unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))sub_97E7A0);
  if ( v6 == *(_QWORD *)(a1 + 184) )
    return 0;
  v7 = *(_QWORD *)(v6 + 8);
  if ( v7 != v9 )
    return 0;
  if ( v7 )
    return memcmp(*(const void **)v6, s2, v7) == 0;
  return 1;
}
