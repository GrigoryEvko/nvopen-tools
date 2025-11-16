// Function: sub_14EE720
// Address: 0x14ee720
//
__int64 *__fastcall sub_14EE720(__int64 *a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  int v5; // eax
  __int64 v6; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v7[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v8; // [rsp+20h] [rbp-40h]
  _QWORD v9[2]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v10; // [rsp+40h] [rbp-20h]

  v6 = a3;
  if ( (unsigned __int64)(a3 - 1) > 0x39 )
  {
    *a4 = 0;
LABEL_3:
    v7[0] = "Unknown attribute kind (";
    v7[1] = &v6;
    v8 = 2819;
    v9[0] = v7;
    v10 = 770;
    v9[1] = ")";
    sub_14EE4B0(a1, a2 + 8, (__int64)v9);
    return a1;
  }
  v5 = byte_4292680[a3 - 1];
  *a4 = v5;
  if ( !v5 )
    goto LABEL_3;
  *a1 = 1;
  return a1;
}
