// Function: sub_39F4380
// Address: 0x39f4380
//
__int64 __fastcall sub_39F4380(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rbx
  __int64 result; // rax

  v3 = a3;
  sub_390D5F0(*(_QWORD *)(a1 + 264), a2, 0);
  switch ( v3 )
  {
    case 0LL:
    case 2LL:
    case 4LL:
    case 5LL:
    case 6LL:
    case 7LL:
    case 10LL:
    case 11LL:
    case 12LL:
    case 13LL:
    case 14LL:
    case 15LL:
    case 16LL:
    case 17LL:
    case 18LL:
    case 19LL:
    case 21LL:
    case 23LL:
      result = 0;
      break;
    case 1LL:
      *(_DWORD *)(a2 + 32) = 0;
      result = 1;
      break;
    case 3LL:
      *(_DWORD *)(a2 + 32) = 1;
      result = 1;
      break;
    case 8LL:
      *(_BYTE *)(a2 + 8) |= 0x10u;
      result = 1;
      break;
    case 9LL:
      *(_BYTE *)(a2 + 37) = 1;
      result = 1;
      break;
    case 20LL:
    case 22LL:
      *(_BYTE *)(a2 + 8) |= 0x10u;
      *(_BYTE *)(a2 + 36) = 1;
      result = 1;
      break;
  }
  return result;
}
