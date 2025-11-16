// Function: sub_388ED60
// Address: 0x388ed60
//
__int64 __fastcall sub_388ED60(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  const char *v4; // rax
  unsigned __int64 v5; // rsi
  _QWORD v6[2]; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+10h] [rbp-30h]
  char v8; // [rsp+11h] [rbp-2Fh]

  if ( (unsigned __int8)sub_388AF10(a1, 342, "expected 'typeTestRes' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_388AF10(a1, 343, "expected 'kind' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
  {
    return 1;
  }
  switch ( *(_DWORD *)(a1 + 64) )
  {
    case 0x158:
      *(_DWORD *)a2 = 0;
      goto LABEL_9;
    case 0x159:
      *(_DWORD *)a2 = 1;
      goto LABEL_9;
    case 0x15A:
      *(_DWORD *)a2 = 2;
      goto LABEL_9;
    case 0x15B:
      *(_DWORD *)a2 = 3;
      goto LABEL_9;
    case 0x15C:
      *(_DWORD *)a2 = 4;
LABEL_9:
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
        || (unsigned __int8)sub_388AF10(a1, 349, "expected 'sizeM1BitWidth' here")
        || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
        || (unsigned __int8)sub_388BA90(a1, (_DWORD *)(a2 + 4)) )
      {
        return 1;
      }
      if ( *(_DWORD *)(a1 + 64) != 4 )
        return sub_388AF10(a1, 13, "expected ')' here");
      break;
    default:
      v8 = 1;
      v4 = "unexpected TypeTestResolution kind";
      goto LABEL_28;
  }
  while ( 1 )
  {
    v3 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v3;
    if ( v3 == 352 )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':'") || (unsigned __int8)sub_388BA90(a1, v6) )
        return 1;
      *(_BYTE *)(a2 + 24) = v6[0];
      goto LABEL_25;
    }
    if ( v3 > 0x160 )
      break;
    if ( v3 == 350 )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':'") || (unsigned __int8)sub_388BD80(a1, (__int64 *)(a2 + 8)) )
        return 1;
    }
    else
    {
      if ( v3 != 351 )
        goto LABEL_36;
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':'")
        || (unsigned __int8)sub_388BD80(a1, (__int64 *)(a2 + 16)) )
      {
        return 1;
      }
    }
LABEL_25:
    if ( *(_DWORD *)(a1 + 64) != 4 )
      return sub_388AF10(a1, 13, "expected ')' here");
  }
  if ( v3 == 353 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':'") || (unsigned __int8)sub_388BD80(a1, (__int64 *)(a2 + 32)) )
      return 1;
    goto LABEL_25;
  }
LABEL_36:
  v8 = 1;
  v4 = "expected optional TypeTestResolution field";
LABEL_28:
  v5 = *(_QWORD *)(a1 + 56);
  v6[0] = v4;
  v7 = 3;
  return sub_38814C0(a1 + 8, v5, (__int64)v6);
}
