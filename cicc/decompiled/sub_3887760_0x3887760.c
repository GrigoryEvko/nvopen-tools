// Function: sub_3887760
// Address: 0x3887760
//
__int64 __fastcall sub_3887760(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v5; // r12d
  int v8; // eax
  _QWORD v9[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v11; // [rsp+20h] [rbp-50h]
  _QWORD v12[2]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v13; // [rsp+40h] [rbp-30h]

  v4 = a1 + 8;
  v5 = *(unsigned __int8 *)(a4 + 1);
  v9[0] = a2;
  v9[1] = a3;
  if ( (_BYTE)v5 )
  {
    v10[0] = "field '";
    v10[1] = v9;
    v12[0] = v10;
    v11 = 1283;
    v12[1] = "' cannot be specified more than once";
    v13 = 770;
    return (unsigned int)sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)v12);
  }
  v8 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v8;
  if ( v8 == 18 )
  {
    *(_WORD *)a4 = 257;
  }
  else
  {
    if ( v8 != 19 )
    {
      v12[0] = "expected 'true' or 'false'";
      v13 = 259;
      return (unsigned int)sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)v12);
    }
    *(_WORD *)a4 = 256;
  }
  *(_DWORD *)(a1 + 64) = sub_3887100(v4);
  return v5;
}
