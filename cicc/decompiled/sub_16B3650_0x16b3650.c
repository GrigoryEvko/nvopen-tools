// Function: sub_16B3650
// Address: 0x16b3650
//
__int64 __fastcall sub_16B3650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _DWORD *a7)
{
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r9
  _QWORD v14[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v15[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v16; // [rsp+20h] [rbp-40h]
  _QWORD v17[2]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v18; // [rsp+40h] [rbp-20h]

  v14[0] = a5;
  v14[1] = a6;
  result = sub_16D2B80(a5, a6, 0, v17);
  if ( !(_BYTE)result && (v11 = v17[0], v17[0] == LODWORD(v17[0])) )
  {
    *a7 = v17[0];
  }
  else
  {
    v12 = sub_16E8CB0(a5, a6, v11);
    v18 = 770;
    v16 = 1283;
    v15[0] = "'";
    v15[1] = v14;
    v17[0] = v15;
    v17[1] = "' value invalid for uint argument!";
    return sub_16B1F90(a2, (__int64)v17, 0, 0, v12, v13);
  }
  return result;
}
