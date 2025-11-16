// Function: sub_8919F0
// Address: 0x8919f0
//
__int64 __fastcall sub_8919F0(__int64 a1, int a2)
{
  __int64 v2; // rcx
  char v3; // dl
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx

  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_BYTE *)(v2 + 80);
  if ( (unsigned __int8)(v3 - 10) <= 1u || (result = 0, v3 == 17) )
  {
    v5 = *(_QWORD *)(v2 + 88);
    result = a2 ? *(_BYTE *)(v5 + 205) & 1 : (unsigned __int8)(*(_BYTE *)(v5 + 192) >> 7);
    if ( !(_DWORD)result
      && (*(_BYTE *)(v5 + 195) & 2) == 0
      && (*(_BYTE *)(v5 + 193) & 0x20) == 0
      && !*(_DWORD *)(v5 + 160) )
    {
      v6 = *(_QWORD *)(v5 + 344);
      if ( !v6 )
      {
        v7 = *(_QWORD *)(a1 + 32);
        switch ( *(_BYTE *)(v7 + 80) )
        {
          case 4:
          case 5:
            v6 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 80LL);
            break;
          case 6:
            v6 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v6 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v6 = *(_QWORD *)(v7 + 88);
            break;
          default:
            break;
        }
        v8 = *(_QWORD *)(v6 + 176);
        result = 1;
        if ( a2 )
        {
          if ( (*(_BYTE *)(v8 + 205) & 1) != 0 )
            return result;
        }
        else if ( *(char *)(v8 + 192) < 0 )
        {
          return result;
        }
        v9 = *(_QWORD *)(v6 + 88);
        if ( !v9 || (*(_BYTE *)(v6 + 160) & 1) != 0 || *(_QWORD *)(v6 + 240) )
          v10 = v6 + 184;
        else
          v10 = *(_QWORD *)(v9 + 88) + 184LL;
        return (*(_BYTE *)(v10 + 64) & 2) != 0;
      }
    }
  }
  return result;
}
