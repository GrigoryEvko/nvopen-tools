// Function: sub_1442290
// Address: 0x1442290
//
__int64 __fastcall sub_1442290(__int64 a1, _QWORD *a2, __int64 *a3)
{
  unsigned int v3; // eax
  unsigned int v4; // r13d
  int v6; // edx
  unsigned __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  char v8; // [rsp+8h] [rbp-28h]

  sub_1441B50((__int64)&v7, a1, *a2 & 0xFFFFFFFFFFFFFFF8LL, a3);
  if ( v8 )
  {
    LOBYTE(v3) = sub_1441D60(a1, v7);
    return v3;
  }
  v4 = sub_1441AE0((_QWORD *)a1);
  if ( !(_BYTE)v4 )
    return v4;
  if ( **(_DWORD **)(a1 + 8) == 1 )
  {
    sub_15E44B0(*(_QWORD *)(*(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL));
    if ( !v6 )
    {
      v4 = (unsigned __int8)byte_4F99E60;
      if ( !byte_4F99E60 )
        return (unsigned int)sub_15602E0(
                               *(_QWORD *)(*(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL) + 112LL,
                               "profile-sample-accurate",
                               23);
    }
    return v4;
  }
  return 0;
}
