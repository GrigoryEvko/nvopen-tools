// Function: sub_E81B70
// Address: 0xe81b70
//
__int64 __fastcall sub_E81B70(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // kr00_8
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi

  v3 = v2;
  result = *(unsigned __int8 *)(a1 + 28);
  switch ( *(_BYTE *)(a1 + 28) )
  {
    case 0:
    case 2:
    case 3:
    case 5:
    case 9:
    case 0xA:
      return result;
    case 1:
      goto LABEL_14;
    case 4:
      v7 = *(_QWORD *)(a1 + 128);
      if ( v7 != a1 + 144 )
        _libc_free(v7, a2);
      goto LABEL_6;
    case 6:
    case 7:
    case 8:
    case 0xD:
LABEL_6:
      v8 = *(_QWORD *)(a1 + 72);
      result = a1 + 88;
      if ( v8 != a1 + 88 )
        goto LABEL_7;
      goto LABEL_8;
    case 0xB:
      v6 = *(_QWORD *)(a1 + 64);
      if ( v6 == a1 + 88 )
        return result;
      goto LABEL_9;
    case 0xC:
      v9 = *(_QWORD *)(a1 + 256);
      if ( v9 != a1 + 280 )
        _libc_free(v9, a2);
      v10 = *(_QWORD *)(a1 + 208);
      if ( v10 != a1 + 224 )
        _libc_free(v10, a2);
LABEL_14:
      v8 = *(_QWORD *)(a1 + 96);
      result = a1 + 112;
      if ( v8 == a1 + 112 )
        goto LABEL_8;
LABEL_7:
      result = _libc_free(v8, a2);
LABEL_8:
      v6 = *(_QWORD *)(a1 + 40);
      if ( v6 != a1 + 64 )
LABEL_9:
        result = _libc_free(v6, a2);
      break;
    default:
      result = v3;
      break;
  }
  return result;
}
