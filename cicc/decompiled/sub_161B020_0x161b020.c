// Function: sub_161B020
// Address: 0x161b020
//
__int64 __fastcall sub_161B020(__int64 a1, int a2, int a3, int a4, int a5, int a6)
{
  __int64 result; // rax
  _BYTE *v7; // rsi
  int v8; // edx
  _BYTE *v9; // rsi
  unsigned __int8 v10; // [rsp+13h] [rbp-25h]
  _DWORD v11[4]; // [rsp+14h] [rbp-24h] BYREF
  _DWORD v12[5]; // [rsp+24h] [rbp-14h] BYREF

  v11[0] = a2;
  v12[0] = 0;
  result = sub_16B3550((int)a1 + 208, a1, a3, a4, a5, a6, (__int64)v12);
  if ( !(_BYTE)result )
  {
    v7 = *(_BYTE **)(a1 + 168);
    if ( v7 == *(_BYTE **)(a1 + 176) )
    {
      sub_B8BD80(a1 + 160, v7, v12);
      result = 0;
    }
    else
    {
      if ( v7 )
      {
        *(_DWORD *)v7 = v12[0];
        v7 = *(_BYTE **)(a1 + 168);
      }
      *(_QWORD *)(a1 + 168) = v7 + 4;
    }
    v8 = v11[0];
    v9 = *(_BYTE **)(a1 + 192);
    *(_DWORD *)(a1 + 16) = v11[0];
    if ( v9 == *(_BYTE **)(a1 + 200) )
    {
      v10 = result;
      sub_B8BBF0(a1 + 184, v9, v11);
      return v10;
    }
    else
    {
      if ( v9 )
      {
        *(_DWORD *)v9 = v8;
        v9 = *(_BYTE **)(a1 + 192);
      }
      *(_QWORD *)(a1 + 192) = v9 + 4;
    }
  }
  return result;
}
