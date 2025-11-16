// Function: sub_17E85A0
// Address: 0x17e85a0
//
__int64 __fastcall sub_17E85A0(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r12d
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  _QWORD *v6; // rsi
  __int64 v7; // rdx
  char v8; // cl
  unsigned __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( byte_4FA5660 )
  {
    v2 = sub_1695A10(a1, 1);
    if ( (_BYTE)v2 )
    {
      v9[0] = *(_QWORD *)(a1 + 48);
      v4 = (_QWORD *)sub_17E8510(a2, v9);
      v6 = v5;
      if ( v4 == v5 )
        return v2;
      while ( 1 )
      {
        v7 = v4[2];
        v8 = *(_BYTE *)(v7 + 16);
        if ( v8 != 1 && (a1 != v7 || v8) )
          break;
        v4 = (_QWORD *)*v4;
        if ( v6 == v4 )
          return v2;
      }
    }
  }
  return 0;
}
