// Function: sub_15F4ED0
// Address: 0x15f4ed0
//
__int64 __fastcall sub_15F4ED0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi

  result = (unsigned int)*(unsigned __int8 *)(a1 + 16) - 24;
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0x18:
    case 0x19:
    case 0x1C:
    case 0x1E:
    case 0x1F:
    case 0x22:
      v4 = a2 + 1;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v5 = *(_QWORD *)(a1 - 8);
      else
        v5 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      result = v5 + 24 * v4;
      if ( !*(_QWORD *)result )
        goto LABEL_7;
      goto LABEL_5;
    case 0x1A:
      result = a1 - 24LL * a2 - 24;
      if ( !*(_QWORD *)result )
        goto LABEL_7;
      goto LABEL_5;
    case 0x1B:
      v9 = 2 * a2 + 1;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        a1 = *(_QWORD *)(a1 - 8);
      else
        a1 -= 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      goto LABEL_17;
    case 0x1D:
      result = a1 + 24LL * a2 - 48;
      if ( !*(_QWORD *)result )
        goto LABEL_7;
      goto LABEL_5;
    case 0x20:
      v9 = 1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
LABEL_17:
      result = a1 + 24 * v9;
      if ( *(_QWORD *)result )
      {
LABEL_5:
        v6 = *(_QWORD *)(result + 8);
        v7 = *(_QWORD *)(result + 16) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v7 = v6;
        if ( v6 )
          *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
      }
LABEL_7:
      *(_QWORD *)result = a3;
      if ( a3 )
      {
        v8 = *(_QWORD *)(a3 + 8);
        *(_QWORD *)(result + 8) = v8;
        if ( v8 )
          *(_QWORD *)(v8 + 16) = (result + 8) | *(_QWORD *)(v8 + 16) & 3LL;
        *(_QWORD *)(result + 16) = (a3 + 8) | *(_QWORD *)(result + 16) & 3LL;
        *(_QWORD *)(a3 + 8) = result;
      }
      break;
    case 0x21:
      if ( *(_QWORD *)(a1 - 24) )
      {
        v10 = *(_QWORD *)(a1 - 16);
        result = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)result = v10;
        if ( v10 )
        {
          result |= *(_QWORD *)(v10 + 16) & 3LL;
          *(_QWORD *)(v10 + 16) = result;
        }
      }
      *(_QWORD *)(a1 - 24) = a3;
      if ( a3 )
      {
        v11 = *(_QWORD *)(a3 + 8);
        *(_QWORD *)(a1 - 16) = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = (a1 - 16) | *(_QWORD *)(v11 + 16) & 3LL;
        v12 = *(_QWORD *)(a1 - 8);
        v13 = a1 - 24;
        result = (a3 + 8) | v12 & 3;
        *(_QWORD *)(v13 + 16) = result;
        *(_QWORD *)(a3 + 8) = v13;
      }
      break;
  }
  return result;
}
