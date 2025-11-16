// Function: sub_21BD570
// Address: 0x21bd570
//
__int64 __fastcall sub_21BD570(
        __int64 a1,
        unsigned __int8 a2,
        int a3,
        int a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10)
{
  __int64 result; // rax
  _DWORD *v11; // rdi
  char v12; // dl
  char v13; // dl
  __int64 v14; // rcx
  char v15; // dl
  char v16; // dl

  result = a1;
  if ( a2 <= 0xAu )
  {
    if ( a2 > 1u )
    {
      switch ( a2 )
      {
        case 2u:
        case 3u:
          *(_BYTE *)(a1 + 4) = 1;
          *(_DWORD *)a1 = a3;
          return result;
        case 4u:
          *(_BYTE *)(a1 + 4) = 1;
          *(_DWORD *)a1 = a4;
          return result;
        case 5u:
          *(_BYTE *)(a1 + 4) = 1;
          *(_DWORD *)a1 = a5;
          return result;
        case 6u:
          v13 = *(_BYTE *)(a6 + 4);
          *(_BYTE *)(a1 + 4) = v13;
          if ( v13 )
            *(_DWORD *)a1 = *(_DWORD *)a6;
          return result;
        case 8u:
          v14 = a7;
          v15 = *(_BYTE *)(a7 + 4);
          *(_BYTE *)(a1 + 4) = v15;
          if ( v15 )
            goto LABEL_13;
          return result;
        case 9u:
          *(_BYTE *)(a1 + 4) = 1;
          *(_DWORD *)a1 = a9;
          return result;
        case 0xAu:
          v14 = a10;
          v16 = *(_BYTE *)(a10 + 4);
          *(_BYTE *)(a1 + 4) = v16;
          if ( !v16 )
            return result;
LABEL_13:
          v11 = (_DWORD *)v14;
          break;
        default:
          goto LABEL_18;
      }
      goto LABEL_14;
    }
    goto LABEL_18;
  }
  if ( a2 != 86 )
  {
LABEL_18:
    *(_BYTE *)(a1 + 4) = 0;
    return result;
  }
  v11 = (_DWORD *)a8;
  v12 = *(_BYTE *)(a8 + 4);
  *(_BYTE *)(result + 4) = v12;
  if ( v12 )
LABEL_14:
    *(_DWORD *)result = *v11;
  return result;
}
