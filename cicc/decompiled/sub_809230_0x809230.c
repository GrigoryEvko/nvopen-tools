// Function: sub_809230
// Address: 0x809230
//
__int64 __fastcall sub_809230(__int64 a1)
{
  __int64 *v1; // rax
  char v2; // cl
  char v3; // dl
  __int64 *v4; // r12
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  char v9; // dl
  _QWORD v10[10]; // [rsp+0h] [rbp-F0h] BYREF
  unsigned int v11; // [rsp+50h] [rbp-A0h]
  int v12; // [rsp+58h] [rbp-98h]

  v1 = sub_746BE0(a1);
  if ( !v1 )
    return 0;
  v2 = *(_BYTE *)(a1 + 186);
  v3 = *((_BYTE *)v1 + 24);
  v4 = v1;
  if ( (v2 & 8) != 0 )
  {
    switch ( v3 )
    {
      case 24:
        result = 0;
        if ( (*(_BYTE *)(a1 + 186) & 2) == 0 )
        {
LABEL_7:
          sub_76C7C0((__int64)v10);
          v10[0] = sub_80A340;
          v10[2] = sub_80A840;
          v11 = 0;
          v12 = 1;
          sub_76CDC0(v4, (__int64)v10, v6, v7, v8);
          return v11;
        }
        break;
      case 5:
        return sub_8DBE70(*v1);
      case 1:
        result = 1;
        if ( *((_BYTE *)v4 + 56) != 105 )
          goto LABEL_7;
        break;
      default:
        goto LABEL_7;
    }
  }
  else
  {
    result = 0;
    if ( v3 == 1 )
    {
      v9 = *((_BYTE *)v4 + 56);
      if ( v9 == 105 )
      {
        return ((*((_BYTE *)v4 + 59) >> 3) ^ 1) & 1;
      }
      else if ( (unsigned __int8)(v9 - 94) <= 1u || (unsigned __int8)(v9 - 100) <= 1u )
      {
        return (v2 & 2) != 0;
      }
    }
  }
  return result;
}
