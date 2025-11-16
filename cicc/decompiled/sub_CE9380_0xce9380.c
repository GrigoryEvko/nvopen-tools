// Function: sub_CE9380
// Address: 0xce9380
//
unsigned __int16 __fastcall sub_CE9380(__int64 a1, int a2)
{
  unsigned __int16 result; // ax
  char *v4; // rsi
  char *v5; // rdi
  char *v6; // rax
  int v7; // edx
  char v8; // cl
  unsigned __int64 v9; // rax
  unsigned __int16 v10; // [rsp+Ch] [rbp-84h]
  __int64 v11; // [rsp+18h] [rbp-78h] BYREF
  char *v12; // [rsp+20h] [rbp-70h] BYREF
  __int64 v13; // [rsp+28h] [rbp-68h]
  _BYTE v14[96]; // [rsp+30h] [rbp-60h] BYREF

  v11 = *(_QWORD *)(a1 + 120);
  v12 = (char *)sub_A74490(&v11, a2);
  result = sub_A73690((__int64 *)&v12);
  if ( !HIBYTE(result) )
  {
    v4 = "align";
    v12 = v14;
    v13 = 0x1000000000LL;
    if ( (unsigned __int8)sub_CE7920(a1, "align", 5u, (__int64)&v12) )
    {
      v5 = v12;
      v4 = &v12[4 * (unsigned int)v13];
      if ( v4 == v12 )
      {
LABEL_14:
        result = 0;
      }
      else
      {
        v6 = v12;
        while ( 1 )
        {
          v7 = *(_DWORD *)v6;
          if ( HIWORD(*(_DWORD *)v6) == a2 )
            break;
          v6 += 4;
          if ( v4 == v6 )
            goto LABEL_14;
        }
        v8 = -1;
        if ( (_WORD)v7 )
        {
          _BitScanReverse64(&v9, (unsigned __int16)v7);
          v8 = 63 - (v9 ^ 0x3F);
        }
        LOBYTE(result) = v8;
        HIBYTE(result) = 1;
      }
    }
    else
    {
      result = 0;
      v5 = v12;
    }
    if ( v5 != v14 )
    {
      v10 = result;
      _libc_free(v5, v4);
      return v10;
    }
  }
  return result;
}
