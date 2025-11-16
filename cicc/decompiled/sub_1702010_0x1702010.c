// Function: sub_1702010
// Address: 0x1702010
//
__int64 __fastcall sub_1702010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax
  __int64 *v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rbx

  result = (unsigned int)*(unsigned __int8 *)(a1 + 16) - 35;
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case '#':
    case '%':
    case '\'':
    case '2':
    case '3':
    case '4':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v7 = *(__int64 **)(a1 - 8);
      else
        v7 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v8 = *v7;
      v9 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v9 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, a5, a6);
        v9 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v9) = v8;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v10 = *(_QWORD *)(a1 - 8);
      else
        v10 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v11 = *(_QWORD *)(v10 + 24);
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)result )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, a5, a6);
        result = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v11;
      ++*(_DWORD *)(a2 + 8);
      break;
    case '$':
    case '&':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '/':
    case '0':
    case '1':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case ':':
    case ';':
    case '<':
    case '=':
    case '>':
      return result;
  }
  return result;
}
