// Function: sub_3112AE0
// Address: 0x3112ae0
//
__int64 __fastcall sub_3112AE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _BYTE *v6; // rdi
  _BYTE *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  char v11; // dl
  char v12; // dl
  char v13; // dl

  v6 = *(_BYTE **)(a1 + 16);
  v7 = *(_BYTE **)(a1 + 8);
  v8 = v6 - v7;
  if ( (unsigned __int64)(v6 - v7) > 8 )
  {
    v6 = v7 + 8;
    v9 = 2;
    goto LABEL_3;
  }
  v9 = v8 >> 2;
  if ( v8 >> 2 )
  {
LABEL_3:
    while ( (unsigned __int8)(*v7 - 32) <= 0x5Eu || (unsigned __int8)(*v7 - 9) <= 4u )
    {
      v11 = v7[1];
      if ( (unsigned __int8)(v11 - 32) > 0x5Eu && (unsigned __int8)(v11 - 9) > 4u )
      {
        LOBYTE(a5) = v6 == v7 + 1;
        return a5;
      }
      v12 = v7[2];
      if ( (unsigned __int8)(v12 - 32) > 0x5Eu && (unsigned __int8)(v12 - 9) > 4u )
      {
        LOBYTE(a5) = v6 == v7 + 2;
        return a5;
      }
      v13 = v7[3];
      if ( (unsigned __int8)(v13 - 32) > 0x5Eu && (unsigned __int8)(v13 - 9) > 4u )
      {
        LOBYTE(a5) = v6 == v7 + 3;
        return a5;
      }
      v7 += 4;
      if ( v9 == 1 )
      {
        v8 = v6 - v7;
        goto LABEL_20;
      }
      v9 = 1;
    }
    goto LABEL_5;
  }
LABEL_20:
  if ( v8 == 2 )
  {
LABEL_28:
    if ( (unsigned __int8)(*v7 - 32) <= 0x5Eu || (unsigned __int8)(*v7 - 9) <= 4u )
    {
      ++v7;
LABEL_23:
      a5 = 1;
      if ( (unsigned __int8)(*v7 - 32) <= 0x5Eu || (unsigned __int8)(*v7 - 9) <= 4u )
        return a5;
      goto LABEL_5;
    }
    goto LABEL_5;
  }
  if ( v8 != 3 )
  {
    a5 = 1;
    if ( v8 != 1 )
      return a5;
    goto LABEL_23;
  }
  if ( (unsigned __int8)(*v7 - 32) <= 0x5Eu || (unsigned __int8)(*v7 - 9) <= 4u )
  {
    ++v7;
    goto LABEL_28;
  }
LABEL_5:
  LOBYTE(a5) = v7 == v6;
  return a5;
}
