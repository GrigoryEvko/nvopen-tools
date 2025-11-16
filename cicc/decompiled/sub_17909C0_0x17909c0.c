// Function: sub_17909C0
// Address: 0x17909c0
//
__int64 __fastcall sub_17909C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v4; // eax
  __int64 result; // rax
  unsigned int v6; // eax
  int v7; // ecx
  unsigned int v8; // eax
  int v9; // ecx

  v3 = *(_QWORD *)a2;
  switch ( *(_BYTE *)(a2 + 16) )
  {
    case '#':
    case '$':
    case '%':
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
    case '3':
    case '4':
      v4 = sub_16431D0(v3);
      *(_DWORD *)(a1 + 8) = v4;
      if ( v4 > 0x40 )
        sub_16A4EF0(a1, 0, 0);
      else
        *(_QWORD *)a1 = 0;
      result = a1;
      break;
    case '\'':
      v8 = sub_16431D0(v3);
      *(_DWORD *)(a1 + 8) = v8;
      if ( v8 <= 0x40 )
      {
        v9 = -v8;
        *(_QWORD *)a1 = (0xFFFFFFFFFFFFFFFFLL >> v9) & 1;
        result = a1;
      }
      else
      {
        sub_16A4EF0(a1, 1, 0);
        result = a1;
      }
      break;
    case '2':
      v6 = sub_16431D0(v3);
      *(_DWORD *)(a1 + 8) = v6;
      if ( v6 > 0x40 )
      {
        sub_16A4EF0(a1, -1, 1);
        result = a1;
      }
      else
      {
        v7 = -v6;
        *(_QWORD *)a1 = 0xFFFFFFFFFFFFFFFFLL >> v7;
        result = a1;
      }
      break;
  }
  return result;
}
