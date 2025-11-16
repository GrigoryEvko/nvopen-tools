// Function: sub_E53FF0
// Address: 0xe53ff0
//
__int64 __fastcall sub_E53FF0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  const char *v6; // rsi
  __int64 v7; // rdi
  __int64 result; // rax
  int i; // eax
  unsigned int v10; // r15d
  int v11; // r12d
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  unsigned int v14; // ecx
  char v15; // [rsp+Fh] [rbp-41h]
  _QWORD v16[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( a3 == 4 )
  {
    v6 = *(const char **)(a1[39] + 240LL);
    if ( !v6 )
      goto LABEL_10;
  }
  else if ( a3 > 4 )
  {
    if ( a3 != 8 )
    {
      if ( (unsigned __int8)sub_E81180(a2, v16) )
      {
        v15 = *(_BYTE *)(a1[39] + 16LL);
LABEL_12:
        for ( i = 0; ; i = v11 )
        {
          v14 = a3 - i;
          if ( a3 - i <= a3 - 1 )
            goto LABEL_13;
          if ( a3 != 1 )
            break;
          v11 = i;
          v10 = 0;
          v12 = 0;
LABEL_14:
          if ( !v15 )
            LOBYTE(i) = a3 - i - v10;
          v13 = sub_E81A90(v12 & (v16[0] >> (8 * (unsigned __int8)i)), a1[1], 0, 0);
          result = sub_E9A5B0(a1, v13, v10, 0);
          if ( a3 == v11 )
            return result;
        }
        v14 = a3 - 1;
LABEL_13:
        _BitScanReverse(&v14, v14);
        v10 = 0x80000000 >> (v14 ^ 0x1F);
        v11 = v10 + i;
        v12 = 0xFFFFFFFFFFFFFFFFLL >> (8 * (8 - (unsigned __int8)v10));
        goto LABEL_14;
      }
LABEL_30:
      sub_C64ED0("Don't know how to emit this value.", 1u);
    }
    v6 = *(const char **)(a1[39] + 248LL);
    if ( !v6 )
    {
LABEL_10:
      if ( (unsigned __int8)sub_E81180(a2, v16) )
      {
        result = *(unsigned __int8 *)(a1[39] + 16LL);
        v15 = *(_BYTE *)(a1[39] + 16LL);
        if ( !a3 )
          return result;
        goto LABEL_12;
      }
      goto LABEL_30;
    }
  }
  else
  {
    if ( a3 != 1 )
    {
      if ( a3 == 2 )
      {
        v6 = *(const char **)(a1[39] + 232LL);
        if ( v6 )
          goto LABEL_6;
      }
      goto LABEL_10;
    }
    v6 = *(const char **)(a1[39] + 224LL);
    if ( !v6 )
      goto LABEL_10;
  }
LABEL_6:
  sub_904010(a1[38], v6);
  v7 = a1[2];
  if ( v7 )
    return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 56LL))(v7, a2);
  sub_E7FAD0(a2, a1[38], a1[39], 0);
  return (__int64)sub_E4D880((__int64)a1);
}
