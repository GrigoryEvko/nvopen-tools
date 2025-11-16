// Function: sub_11013B0
// Address: 0x11013b0
//
unsigned __int8 *__fastcall sub_11013B0(const __m128i *a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned __int8 *result; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // eax
  char v7; // [rsp+3h] [rbp-6Dh]
  unsigned int v8; // [rsp+4h] [rbp-6Ch]
  int v9; // [rsp+4h] [rbp-6Ch]
  __int64 v10; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v11; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v12; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v13; // [rsp+8h] [rbp-68h]
  _BYTE v14[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  v2 = *(_QWORD *)(a2 - 32);
  if ( (unsigned __int8)(*(_BYTE *)v2 - 72) > 1u )
    return 0;
  v4 = *(_QWORD *)(v2 - 32);
  v5 = *(_QWORD *)(a2 + 8);
  v10 = *(_QWORD *)(v4 + 8);
  v7 = *(_BYTE *)a2;
  if ( !sub_10FD370((char *)v2, a1) )
  {
    v9 = sub_BCB060(v5);
    if ( (int)sub_BCB090(*(_QWORD *)(v2 + 8)) < v9 )
      return 0;
  }
  v8 = sub_BCB060(v5);
  v6 = sub_BCB060(v10);
  if ( v8 <= v6 )
  {
    if ( v8 >= v6 )
    {
      return sub_F162A0((__int64)a1, a2, v4);
    }
    else
    {
      v15 = 257;
      result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
      if ( result )
      {
        v13 = result;
        sub_B51510((__int64)result, v4, v5, (__int64)v14, 0, 0);
        return v13;
      }
    }
  }
  else if ( *(_BYTE *)v2 == 73 && v7 == 71 )
  {
    v15 = 257;
    result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
    if ( result )
    {
      v11 = result;
      sub_B51650((__int64)result, v4, v5, (__int64)v14, 0, 0);
      return v11;
    }
  }
  else
  {
    v15 = 257;
    result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
    if ( result )
    {
      v12 = result;
      sub_B515B0((__int64)result, v4, v5, (__int64)v14, 0, 0);
      return v12;
    }
  }
  return result;
}
