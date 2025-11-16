// Function: sub_2C0D840
// Address: 0x2c0d840
//
__int64 **__fastcall sub_2C0D840(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 **v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 **v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 **result; // rax
  char v13; // r8
  char v14; // r8
  bool v15; // zf
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a1;
  v5 = (a2 - (__int64)a1) >> 5;
  v6 = (a2 - (__int64)a1) >> 3;
  v16[0] = a3;
  if ( v5 <= 0 )
  {
LABEL_19:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          return (__int64 **)a2;
LABEL_27:
        v15 = (unsigned __int8)sub_2C0D5F0(v16, *v4) == 0;
        result = v4;
        if ( v15 )
          return (__int64 **)a2;
        return result;
      }
      v13 = sub_2C0D5F0(v16, *v4);
      result = v4;
      if ( v13 )
        return result;
      ++v4;
    }
    v14 = sub_2C0D5F0(v16, *v4);
    result = v4;
    if ( v14 )
      return result;
    ++v4;
    goto LABEL_27;
  }
  v7 = &a1[4 * v5];
  while ( 1 )
  {
    v11 = v16[0];
    if ( v16[0] )
      v11 = v16[0] + 96;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, __int64))(**v4 + 24))(*v4, v11) )
      return v4;
    v8 = v16[0];
    if ( v16[0] )
      v8 = v16[0] + 96;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, __int64))(*v4[1] + 24))(v4[1], v8) )
      return v4 + 1;
    v9 = v16[0];
    if ( v16[0] )
      v9 = v16[0] + 96;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, __int64))(*v4[2] + 24))(v4[2], v9) )
      return v4 + 2;
    v10 = v16[0];
    if ( v16[0] )
      v10 = v16[0] + 96;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, __int64))(*v4[3] + 24))(v4[3], v10) )
      return v4 + 3;
    v4 += 4;
    if ( v4 == v7 )
    {
      v6 = (a2 - (__int64)v4) >> 3;
      goto LABEL_19;
    }
  }
}
