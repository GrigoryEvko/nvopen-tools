// Function: sub_38A2140
// Address: 0x38a2140
//
__int64 __fastcall sub_38A2140(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  int v6; // eax
  __int64 result; // rax
  int v8; // eax
  __int64 v9[2]; // [rsp+0h] [rbp-30h] BYREF
  char v10; // [rsp+10h] [rbp-20h]
  char v11; // [rsp+11h] [rbp-1Fh]

  v6 = *(_DWORD *)(a1 + 64);
  if ( v6 == 376 )
  {
    result = sub_38A9970(a1, v9, 0);
    if ( !(_BYTE)result )
      goto LABEL_7;
  }
  else
  {
    if ( v6 != 14 )
    {
      v11 = 1;
      v9[0] = (__int64)"expected metadata operand";
      v10 = 3;
      return sub_38A2070(a1, a2, (__int64)v9, a3, a4, a5, a6);
    }
    v8 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v8;
    if ( v8 == 377 )
    {
      result = sub_388B260((__int64 **)a1, v9);
      if ( !(_BYTE)result )
        goto LABEL_7;
    }
    else
    {
      result = sub_38A2440(a1, v9);
      if ( !(_BYTE)result )
LABEL_7:
        *a2 = v9[0];
    }
  }
  return result;
}
