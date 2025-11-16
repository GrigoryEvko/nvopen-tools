// Function: sub_393EF60
// Address: 0x393ef60
//
__int64 __fastcall sub_393EF60(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // [rsp+0h] [rbp-50h] BYREF
  __int64 v3; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v4[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v5[2]; // [rsp+20h] [rbp-30h] BYREF
  __int64 v6; // [rsp+30h] [rbp-20h] BYREF
  _BYTE *v7; // [rsp+40h] [rbp-10h]
  __int64 v8; // [rsp+48h] [rbp-8h]

  sub_16F4AE0((__int64)&v6, a1, 1, 35);
  result = 0;
  if ( v6 )
  {
    if ( *v7 != 32 )
    {
      v5[0] = v7;
      v4[0] = 0;
      v4[1] = 0;
      v5[1] = v8;
      return sub_393DBD0(v5, v4, &v2, &v3);
    }
  }
  return result;
}
