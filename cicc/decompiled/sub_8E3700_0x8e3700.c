// Function: sub_8E3700
// Address: 0x8e3700
//
__int64 __fastcall sub_8E3700(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl
  __int64 v3; // rax
  bool v4; // zf
  const char *v5; // rdx
  char v6; // dl
  __int64 v7; // rax
  const char *v8; // rdx
  char v9; // dl
  __int64 v10; // rax
  const char *v11; // rdx
  char v12; // dl
  __int64 v13; // rax
  const char *v14; // rdx
  _BYTE v15[9]; // [rsp+Fh] [rbp-11h] BYREF

  result = sub_8D9610(a1, v15);
  if ( (_DWORD)result )
  {
    result = v15[0];
    if ( unk_4D04710 && (v15[0] & 1) != 0 )
    {
      v2 = *(_BYTE *)(a1 + 140);
      if ( v2 == 12 )
      {
        v3 = a1;
        do
        {
          v3 = *(_QWORD *)(v3 + 160);
          v2 = *(_BYTE *)(v3 + 140);
        }
        while ( v2 == 12 );
      }
      v4 = v2 == 2;
      v5 = "is";
      if ( !v4 )
        v5 = "contains";
      sub_6870B0(0xDEBu, dword_4F07508, (__int64)v5, (__int64)"128-bit integer", a1);
      result = v15[0];
    }
    if ( (result & 2) != 0 )
    {
      v6 = *(_BYTE *)(a1 + 140);
      if ( v6 == 12 )
      {
        v7 = a1;
        do
        {
          v7 = *(_QWORD *)(v7 + 160);
          v6 = *(_BYTE *)(v7 + 140);
        }
        while ( v6 == 12 );
      }
      v4 = v6 == 15;
      v8 = "is";
      if ( !v4 )
        v8 = "contains";
      sub_6870B0(0xDEBu, dword_4F07508, (__int64)v8, (__int64)"vector", a1);
      result = v15[0];
    }
    if ( unk_4D0470C && (result & 4) != 0 )
    {
      v9 = *(_BYTE *)(a1 + 140);
      if ( v9 == 12 )
      {
        v10 = a1;
        do
        {
          v10 = *(_QWORD *)(v10 + 160);
          v9 = *(_BYTE *)(v10 + 140);
        }
        while ( v9 == 12 );
      }
      v4 = v9 == 3;
      v11 = "is";
      if ( !v4 )
        v11 = "contains";
      sub_6870B0(0xDEBu, dword_4F07508, (__int64)v11, (__int64)"128-bit floating-point", a1);
      result = v15[0];
    }
    if ( (result & 8) != 0 )
    {
      v12 = *(_BYTE *)(a1 + 140);
      if ( v12 == 12 )
      {
        v13 = a1;
        do
        {
          v13 = *(_QWORD *)(v13 + 160);
          v12 = *(_BYTE *)(v13 + 140);
        }
        while ( v12 == 12 );
      }
      v4 = v12 == 5;
      v14 = "is";
      if ( !v4 )
        v14 = "contains";
      return sub_6870B0(0xDEBu, dword_4F07508, (__int64)v14, (__int64)"_Complex", a1);
    }
  }
  return result;
}
