// Function: sub_1F40B60
// Address: 0x1f40b60
//
__int64 __fastcall sub_1F40B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 (__fastcall *v8)(__int64, __int64); // rax
  unsigned int v9; // eax
  unsigned __int8 v10; // dl
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  __int64 v12; // [rsp+8h] [rbp-28h]

  v11 = a2;
  v12 = a3;
  if ( (_BYTE)a2 )
  {
    if ( (unsigned __int8)(a2 - 14) <= 0x5Fu )
      return v11;
  }
  else if ( (unsigned __int8)sub_1F58D20(&v11) )
  {
    return v11;
  }
  if ( a5 && (v8 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL), v8 != sub_1F3D8E0) )
  {
    return ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, __int64))v8)(a1, a4, (unsigned int)v11, v12);
  }
  else
  {
    v9 = 8 * sub_15A9520(a4, 0);
    if ( v9 == 32 )
    {
      return 5;
    }
    else if ( v9 > 0x20 )
    {
      v10 = 6;
      if ( v9 != 64 )
      {
        v10 = 0;
        if ( v9 == 128 )
          return 7;
      }
    }
    else
    {
      v10 = 3;
      if ( v9 != 8 )
        return (unsigned __int8)(4 * (v9 == 16));
    }
  }
  return v10;
}
