// Function: sub_18F2D80
// Address: 0x18f2d80
//
__int64 __fastcall sub_18F2D80(__int64 a1)
{
  unsigned __int8 v1; // al
  unsigned int v2; // r8d
  __int64 result; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rdi
  unsigned int v8; // ebx

  v1 = *(_BYTE *)(a1 + 16);
  v2 = 0;
  if ( v1 <= 0x17u )
    return v2;
  if ( v1 == 78 )
  {
    v4 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v4 + 16) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
    {
      switch ( *(_DWORD *)(v4 + 36) )
      {
        case 0x6D:
        case 0x6E:
        case 0x6F:
        case 0x70:
        case 0x71:
        case 0x72:
        case 0x73:
        case 0x75:
        case 0x76:
        case 0x77:
        case 0x78:
        case 0x79:
        case 0x7A:
        case 0x7B:
        case 0x7C:
        case 0x7D:
        case 0x7E:
        case 0x7F:
        case 0x80:
        case 0x81:
        case 0x82:
        case 0x83:
        case 0x84:
        case 0x86:
        case 0x88:
        case 0x8A:
          return 1;
        case 0x74:
          return 0;
        case 0x85:
        case 0x87:
        case 0x89:
          v7 = *(_QWORD *)(a1 + 24 * (3LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
          v8 = *(_DWORD *)(v7 + 32);
          if ( v8 <= 0x40 )
            LOBYTE(v2) = *(_QWORD *)(v7 + 24) == 0;
          else
            LOBYTE(v2) = v8 == (unsigned int)sub_16A57B0(v7 + 24);
          result = v2;
          break;
      }
      return result;
    }
    v5 = a1 | 4;
  }
  else
  {
    if ( v1 != 29 )
      return v2;
    v5 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v6 )
    return 0;
  LOBYTE(v2) = *(_QWORD *)(v6 + 8) == 0;
  return v2;
}
