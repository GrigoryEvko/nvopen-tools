// Function: sub_2BDFB70
// Address: 0x2bdfb70
//
char *__fastcall sub_2BDFB70(__int64 a1)
{
  char *result; // rax
  char *v2; // rcx
  char v4; // r8
  __int64 (__fastcall *v5)(_QWORD); // rax
  _QWORD *v6; // rdi
  char v7; // dl
  __int64 v8; // rdx
  __int64 v9; // rdx

  result = *(char **)(a1 + 176);
  v2 = *(char **)(a1 + 184);
  if ( result == v2 )
    goto LABEL_25;
  *(_QWORD *)(a1 + 176) = result + 1;
  v4 = *result;
  if ( *result == 45 )
  {
    *(_DWORD *)(a1 + 144) = 28;
    *(_BYTE *)(a1 + 168) = 0;
    return result;
  }
  if ( v4 == 91 )
  {
    if ( v2 != result + 1 )
    {
      v7 = result[1];
      switch ( v7 )
      {
        case '.':
          *(_DWORD *)(a1 + 144) = 16;
          break;
        case ':':
          *(_DWORD *)(a1 + 144) = 15;
          break;
        case '=':
          *(_DWORD *)(a1 + 144) = 17;
          break;
        default:
          v8 = *(_QWORD *)(a1 + 208);
          *(_DWORD *)(a1 + 144) = 1;
          result = (char *)sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, v8, 1u, 91);
LABEL_18:
          *(_BYTE *)(a1 + 168) = 0;
          return result;
      }
      *(_QWORD *)(a1 + 176) = result + 2;
      result = sub_2BDFA30((_QWORD *)a1, result[1]);
      *(_BYTE *)(a1 + 168) = 0;
      return result;
    }
LABEL_25:
    abort();
  }
  if ( v4 == 93 )
  {
    if ( (*(_BYTE *)(a1 + 140) & 0x10) != 0 || !*(_BYTE *)(a1 + 168) )
    {
      *(_DWORD *)(a1 + 144) = 11;
      *(_DWORD *)(a1 + 136) = 0;
      *(_BYTE *)(a1 + 168) = 0;
      return result;
    }
    goto LABEL_17;
  }
  if ( v4 != 92 || (*(_BYTE *)(a1 + 140) & 0x90) == 0 )
  {
LABEL_17:
    v9 = *(_QWORD *)(a1 + 208);
    *(_DWORD *)(a1 + 144) = 1;
    result = (char *)sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, v9, 1u, v4);
    goto LABEL_18;
  }
  v5 = *(__int64 (__fastcall **)(_QWORD))(a1 + 232);
  v6 = (_QWORD *)(a1 + *(_QWORD *)(a1 + 240));
  if ( ((unsigned __int8)v5 & 1) != 0 )
    v5 = *(__int64 (__fastcall **)(_QWORD))((char *)v5 + *v6 - 1);
  result = (char *)v5(v6);
  *(_BYTE *)(a1 + 168) = 0;
  return result;
}
