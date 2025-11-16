// Function: sub_2537310
// Address: 0x2537310
//
__int64 __fastcall sub_2537310(_BYTE *a1, __int64 a2)
{
  __int64 result; // rax
  char v3; // [rsp+Bh] [rbp-35h] BYREF
  unsigned int v4; // [rsp+Ch] [rbp-34h] BYREF
  _QWORD v5[5]; // [rsp+10h] [rbp-30h] BYREF

  v5[1] = a2;
  v4 = 1;
  v5[0] = &v4;
  v5[2] = a1;
  v3 = 1;
  if ( (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_257AB00,
                          (__int64)v5,
                          (__int64)a1,
                          1u,
                          &v3) )
  {
    result = v4;
    if ( !v4 && a1[97] != 3 && a1[96] != 3 && a1[99] != 3 && a1[98] != 3 )
      a1[100] = 1;
  }
  else
  {
    result = (unsigned __int8)a1[100];
    a1[100] = 1;
  }
  return result;
}
