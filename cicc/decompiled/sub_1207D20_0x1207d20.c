// Function: sub_1207D20
// Address: 0x1207d20
//
__int64 __fastcall sub_1207D20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v5; // r13d
  unsigned __int64 v6; // rsi
  int v9; // eax
  unsigned __int64 v10; // rsi
  _QWORD v11[4]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v12; // [rsp+20h] [rbp-60h]
  _QWORD v13[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v14; // [rsp+50h] [rbp-30h]

  v4 = a1 + 176;
  v5 = *(unsigned __int8 *)(a4 + 1);
  if ( (_BYTE)v5 )
  {
    v11[2] = a2;
    v11[0] = "field '";
    v14 = 770;
    v6 = *(_QWORD *)(a1 + 232);
    v12 = 1283;
    v11[3] = a3;
    v13[0] = v11;
    v13[2] = "' cannot be specified more than once";
    sub_11FD800(a1 + 176, v6, (__int64)v13, 1);
    return v5;
  }
  v9 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v9;
  if ( v9 == 20 )
  {
    *(_WORD *)a4 = 257;
  }
  else
  {
    if ( v9 != 21 )
    {
      v10 = *(_QWORD *)(a1 + 232);
      v14 = 259;
      v13[0] = "expected 'true' or 'false'";
      sub_11FD800(v4, v10, (__int64)v13, 1);
      return 1;
    }
    *(_WORD *)a4 = 256;
  }
  *(_DWORD *)(a1 + 240) = sub_1205200(v4);
  return v5;
}
