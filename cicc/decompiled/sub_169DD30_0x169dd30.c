// Function: sub_169DD30
// Address: 0x169dd30
//
__int64 __fastcall sub_169DD30(__int16 **a1, _BYTE *a2, _BYTE *a3, unsigned int a4)
{
  __int64 result; // rax
  char v7; // al
  char v8; // al
  unsigned int v9; // r13d
  unsigned __int8 v10; // dl

  if ( (unsigned __int8)sub_169CC60(a1) || (unsigned __int8)sub_169CC60(a3) || (unsigned __int8)sub_169CC60(a2) )
  {
    sub_16986F0(a1, 0, 0, 0);
    return 1;
  }
  else
  {
    v7 = *((_BYTE *)a1 + 18) & 0xF7 | (*((_BYTE *)a1 + 18) ^ a2[18]) & 8;
    *((_BYTE *)a1 + 18) = v7;
    if ( (v7 & 6) != 0 && (v7 & 7) != 3 && (v8 = a2[18], (v8 & 6) != 0) && (v8 & 7) != 3 && (a3[18] & 6) != 0 )
    {
      v9 = sub_16999D0((__int64)a1, (__int64)a2, (__int64)a3);
      result = sub_1698EC0(a1, a4, v9);
      if ( v9 )
        result = (unsigned int)result | 0x10;
      v10 = *((_BYTE *)a1 + 18);
      if ( (v10 & 7) == 3 && (result & 8) == 0 && ((v10 ^ a3[18]) & 8) != 0 )
        *((_BYTE *)a1 + 18) = (8 * (a4 == 2)) | v10 & 0xF7;
    }
    else
    {
      result = sub_169DBD0(a1, a2);
      if ( !(_DWORD)result )
        return sub_169CDE0(a1, a3, a4, 0);
    }
  }
  return result;
}
