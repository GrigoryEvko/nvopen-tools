// Function: sub_E207A0
// Address: 0xe207a0
//
__int64 __fastcall sub_E207A0(size_t *a1)
{
  char *v2; // rdx
  char v3; // al
  __int64 v4; // rax
  char v5; // cl
  unsigned int v6; // edx
  int v7; // [rsp+0h] [rbp-20h]

  if ( (unsigned __int8)sub_E20730(a1, 3u, &unk_3F7C290) )
    return 0x300000000LL;
  v2 = (char *)a1[1];
  v3 = *v2;
  --*a1;
  a1[1] = (size_t)(v2 + 1);
  switch ( v3 )
  {
    case 'A':
      v6 = 0;
      v4 = 2;
      v5 = 0;
      break;
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'S':
      BYTE2(v7) = 0;
      v5 = 3;
      LOWORD(v7) = 0;
      v4 = 1;
      v6 = v7 << 8;
      break;
    case 'P':
      v6 = 0;
      v4 = 1;
      v5 = 0;
      break;
    case 'Q':
      BYTE2(v7) = 0;
      v4 = 1;
      LOWORD(v7) = 0;
      v5 = 1;
      v6 = v7 << 8;
      break;
    case 'R':
      BYTE2(v7) = 0;
      v4 = 1;
      v5 = 2;
      LOWORD(v7) = 0;
      v6 = v7 << 8;
      break;
  }
  LOBYTE(v6) = v5;
  return (v4 << 32) | v6;
}
