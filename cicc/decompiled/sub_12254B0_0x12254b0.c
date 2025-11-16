// Function: sub_12254B0
// Address: 0x12254b0
//
__int64 __fastcall sub_12254B0(__int64 a1, _QWORD *a2, __int64 *a3)
{
  int v4; // eax
  __int64 result; // rax
  int v6; // eax
  __int64 v7[4]; // [rsp+0h] [rbp-50h] BYREF
  char v8; // [rsp+20h] [rbp-30h]
  char v9; // [rsp+21h] [rbp-2Fh]

  v4 = *(_DWORD *)(a1 + 240);
  if ( v4 == 511 )
  {
    if ( (unsigned int)sub_2241AC0(a1 + 248, "DIArgList") )
    {
      result = sub_122E1E0(a1, v7, 0);
      if ( !(_BYTE)result )
        goto LABEL_7;
    }
    else
    {
      result = sub_12252E0(a1, v7, a3);
      if ( !(_BYTE)result )
        goto LABEL_7;
    }
  }
  else
  {
    if ( v4 != 14 )
    {
      v9 = 1;
      v7[0] = (__int64)"expected metadata operand";
      v8 = 3;
      return sub_1225220(a1, a2, (int *)v7, a3);
    }
    v6 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v6;
    if ( v6 == 512 )
    {
      result = sub_120B460((__int64 **)a1, v7);
      if ( !(_BYTE)result )
        goto LABEL_7;
    }
    else
    {
      result = sub_1225820(a1, v7);
      if ( !(_BYTE)result )
LABEL_7:
        *a2 = v7[0];
    }
  }
  return result;
}
