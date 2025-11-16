// Function: sub_9C8460
// Address: 0x9c8460
//
__int64 *__fastcall sub_9C8460(__int64 *a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  int v5; // eax
  __int64 v6; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v7[4]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v8; // [rsp+30h] [rbp-50h]
  _QWORD v9[4]; // [rsp+40h] [rbp-40h] BYREF
  __int16 v10; // [rsp+60h] [rbp-20h]

  v6 = a3;
  if ( (unsigned __int64)(a3 - 1) > 0x65 )
  {
    *a4 = 0;
LABEL_3:
    v7[0] = "Unknown attribute kind (";
    v7[2] = &v6;
    v8 = 2819;
    v9[0] = v7;
    v10 = 770;
    v9[2] = ")";
    sub_9C81F0(a1, a2 + 8, (__int64)v9);
    return a1;
  }
  v5 = byte_3F221A0[a3 - 1];
  *a4 = v5;
  if ( !v5 )
    goto LABEL_3;
  *a1 = 1;
  return a1;
}
