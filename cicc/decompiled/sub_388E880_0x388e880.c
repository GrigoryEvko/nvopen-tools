// Function: sub_388E880
// Address: 0x388e880
//
__int64 __fastcall sub_388E880(__int64 a1, _DWORD *a2, int a3)
{
  __int64 v4; // rdi
  unsigned int v5; // eax
  const char *v7; // rax
  unsigned __int64 v8; // rsi
  const char *v9; // [rsp+0h] [rbp-30h] BYREF
  char v10; // [rsp+10h] [rbp-20h]
  char v11; // [rsp+11h] [rbp-1Fh]

  v4 = a1 + 8;
  v5 = *(_DWORD *)(v4 + 56);
  if ( a3 != 52 )
  {
    switch ( v5 )
    {
      case 0xCFu:
        *a2 = 32;
        goto LABEL_9;
      case 0xD0u:
        *a2 = 33;
        goto LABEL_9;
      case 0xD1u:
        *a2 = 40;
        goto LABEL_9;
      case 0xD2u:
        *a2 = 38;
        goto LABEL_9;
      case 0xD3u:
        *a2 = 41;
        goto LABEL_9;
      case 0xD4u:
        *a2 = 39;
        goto LABEL_9;
      case 0xD5u:
        *a2 = 36;
        goto LABEL_9;
      case 0xD6u:
        *a2 = 34;
        goto LABEL_9;
      case 0xD7u:
        *a2 = 37;
        goto LABEL_9;
      case 0xD8u:
        *a2 = 35;
        goto LABEL_9;
      default:
        v11 = 1;
        v7 = "expected icmp predicate (e.g. 'eq')";
        goto LABEL_21;
    }
  }
  if ( v5 <= 0xE2 )
  {
    if ( v5 > 0xD4 )
    {
      switch ( v5 )
      {
        case 0xD6u:
          *a2 = 10;
          break;
        case 0xD7u:
          *a2 = 13;
          break;
        case 0xD8u:
          *a2 = 11;
          break;
        case 0xD9u:
          *a2 = 1;
          break;
        case 0xDAu:
          *a2 = 6;
          break;
        case 0xDBu:
          *a2 = 4;
          break;
        case 0xDCu:
          *a2 = 2;
          break;
        case 0xDDu:
          *a2 = 5;
          break;
        case 0xDEu:
          *a2 = 3;
          break;
        case 0xDFu:
          *a2 = 7;
          break;
        case 0xE0u:
          *a2 = 8;
          break;
        case 0xE1u:
          *a2 = 9;
          break;
        case 0xE2u:
          *a2 = 14;
          break;
        default:
          *a2 = 12;
          break;
      }
      goto LABEL_9;
    }
    if ( v5 == 18 )
    {
      *a2 = 15;
      goto LABEL_9;
    }
    if ( v5 == 19 )
    {
      *a2 = 0;
LABEL_9:
      *(_DWORD *)(a1 + 64) = sub_3887100(v4);
      return 0;
    }
  }
  v11 = 1;
  v7 = "expected fcmp predicate (e.g. 'oeq')";
LABEL_21:
  v8 = *(_QWORD *)(a1 + 56);
  v9 = v7;
  v10 = 3;
  return sub_38814C0(v4, v8, (__int64)&v9);
}
