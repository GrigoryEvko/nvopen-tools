// Function: sub_1225CE0
// Address: 0x1225ce0
//
__int64 __fastcall sub_1225CE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  __int64 result; // rax
  __int64 v9; // rdx
  _QWORD v10[4]; // [rsp+0h] [rbp-70h] BYREF
  __int16 v11; // [rsp+20h] [rbp-50h]
  _QWORD v12[4]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v13; // [rsp+50h] [rbp-20h]

  if ( *(_DWORD *)(a1 + 240) == 54 )
  {
    v6 = a1 + 176;
    if ( *(_BYTE *)(a4 + 9) )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v6);
      *(_BYTE *)(a4 + 8) = 1;
      *(_QWORD *)a4 = 0;
      return 0;
    }
    else
    {
      v10[2] = a2;
      v7 = *(_QWORD *)(a1 + 232);
      v11 = 1283;
      v10[0] = "'";
      v10[3] = a3;
      v12[0] = v10;
      v13 = 770;
      v12[2] = "' cannot be null";
      sub_11FD800(v6, v7, (__int64)v12, 1);
      return 1;
    }
  }
  else
  {
    result = sub_12254B0(a1, v12, 0);
    if ( !(_BYTE)result )
    {
      v9 = v12[0];
      *(_BYTE *)(a4 + 8) = 1;
      *(_QWORD *)a4 = v9;
    }
  }
  return result;
}
