// Function: sub_C55B80
// Address: 0xc55b80
//
__int64 __fastcall sub_C55B80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _QWORD *a7)
{
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  _QWORD v18[4]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v19; // [rsp+20h] [rbp-60h]
  _QWORD v20[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v21; // [rsp+50h] [rbp-30h]

  result = sub_C93C90(a5, a6, 0, v20);
  if ( (_BYTE)result )
  {
    v17 = sub_CEADF0(a5, a6, v13, v14, v15, v16);
    v21 = 770;
    v19 = 1283;
    v18[0] = "'";
    v20[0] = v18;
    v18[2] = a5;
    v18[3] = a6;
    v20[2] = "' value invalid for ulong argument!";
    return sub_C53280(a2, (__int64)v20, 0, 0, v17);
  }
  else
  {
    *a7 = v20[0];
  }
  return result;
}
