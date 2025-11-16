// Function: sub_388F140
// Address: 0x388f140
//
char __fastcall sub_388F140(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r14
  char v3; // bl
  char result; // al
  unsigned int v5; // eax
  bool v6; // cc
  unsigned __int64 v7; // rsi
  _QWORD v8[2]; // [rsp+0h] [rbp-40h] BYREF
  char v9; // [rsp+10h] [rbp-30h]
  char v10; // [rsp+11h] [rbp-2Fh]

  v2 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v3 = sub_388AF10(a1, 16, "expected ':' in funcFlags");
  result = v3 | sub_388AF10(a1, 12, "expected '(' in funcFlags");
  if ( !result )
  {
    v5 = *(_DWORD *)(a1 + 64);
    v6 = v5 <= 0x13F;
    if ( v5 == 319 )
      goto LABEL_10;
LABEL_3:
    if ( v6 )
    {
      if ( v5 == 317 )
      {
        *(_DWORD *)(a1 + 64) = sub_3887100(v2);
        if ( !(unsigned __int8)sub_388AF10(a1, 16, "expected ':'") && !(unsigned __int8)sub_388F090(a1, v8) )
        {
          *a2 = v8[0] & 1 | *a2 & 0xFE;
LABEL_17:
          while ( *(_DWORD *)(a1 + 64) == 4 )
          {
LABEL_9:
            v5 = sub_3887100(v2);
            *(_DWORD *)(a1 + 64) = v5;
            v6 = v5 <= 0x13F;
            if ( v5 != 319 )
              goto LABEL_3;
LABEL_10:
            *(_DWORD *)(a1 + 64) = sub_3887100(v2);
            if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':'") || (unsigned __int8)sub_388F090(a1, v8) )
              return 1;
            *a2 = (4 * (v8[0] & 1)) | *a2 & 0xFB;
          }
          return sub_388AF10(a1, 13, "expected ')' in funcFlags");
        }
        return 1;
      }
      if ( v5 == 318 )
      {
        *(_DWORD *)(a1 + 64) = sub_3887100(v2);
        if ( !(unsigned __int8)sub_388AF10(a1, 16, "expected ':'") && !(unsigned __int8)sub_388F090(a1, v8) )
        {
          *a2 = (2 * (v8[0] & 1)) | *a2 & 0xFD;
          if ( *(_DWORD *)(a1 + 64) == 4 )
            goto LABEL_9;
          return sub_388AF10(a1, 13, "expected ')' in funcFlags");
        }
        return 1;
      }
    }
    else if ( v5 == 320 )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(v2);
      if ( !(unsigned __int8)sub_388AF10(a1, 16, "expected ':'") && !(unsigned __int8)sub_388F090(a1, v8) )
      {
        *a2 = (8 * (v8[0] & 1)) | *a2 & 0xF7;
        goto LABEL_17;
      }
      return 1;
    }
    v7 = *(_QWORD *)(a1 + 56);
    v10 = 1;
    v9 = 3;
    v8[0] = "expected function flag type";
    return sub_38814C0(v2, v7, (__int64)v8);
  }
  return result;
}
