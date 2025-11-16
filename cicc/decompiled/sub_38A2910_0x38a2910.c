// Function: sub_38A2910
// Address: 0x38a2910
//
__int64 __fastcall sub_38A2910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  bool v9; // zf
  __int64 v10; // rdi
  unsigned __int64 v11; // rsi
  __int64 result; // rax
  __int64 v13; // rdx
  _QWORD v14[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v15[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v16; // [rsp+20h] [rbp-40h]
  _QWORD v17[2]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v18; // [rsp+40h] [rbp-20h]

  v9 = *(_DWORD *)(a1 + 64) == 51;
  v14[0] = a2;
  v14[1] = a3;
  if ( v9 )
  {
    v10 = a1 + 8;
    if ( *(_BYTE *)(a4 + 9) )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(v10);
      *(_BYTE *)(a4 + 8) = 1;
      *(_QWORD *)a4 = 0;
      return 0;
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 56);
      v16 = 1283;
      v15[0] = "'";
      v15[1] = v14;
      v17[0] = v15;
      v18 = 770;
      v17[1] = "' cannot be null";
      return sub_38814C0(v10, v11, (__int64)v17);
    }
  }
  else
  {
    result = sub_38A2140(a1, v17, 0, a5, a6, a7);
    if ( !(_BYTE)result )
    {
      v13 = v17[0];
      *(_BYTE *)(a4 + 8) = 1;
      *(_QWORD *)a4 = v13;
    }
  }
  return result;
}
