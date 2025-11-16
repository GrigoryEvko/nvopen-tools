// Function: sub_12082C0
// Address: 0x12082c0
//
__int64 __fastcall sub_12082C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  unsigned int v6; // r14d
  unsigned __int64 v9; // rsi
  _QWORD v10[4]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v11; // [rsp+20h] [rbp-60h]
  _QWORD v12[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v13; // [rsp+50h] [rbp-30h]

  v5 = a1 + 176;
  v6 = *(unsigned __int8 *)(a4 + 8);
  if ( (_BYTE)v6 )
  {
    v9 = *(_QWORD *)(a1 + 232);
    v11 = 1283;
    v10[0] = "field '";
    v10[2] = "column";
    v12[0] = v10;
    v13 = 770;
    v12[2] = "' cannot be specified more than once";
    v10[3] = 6;
    sub_11FD800(v5, v9, (__int64)v12, 1);
    return v6;
  }
  else
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(v5);
    return sub_1208110(a1, (__int64)"column", 6, a4);
  }
}
