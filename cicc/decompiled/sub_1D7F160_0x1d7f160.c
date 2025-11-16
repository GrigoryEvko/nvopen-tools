// Function: sub_1D7F160
// Address: 0x1d7f160
//
__int64 __fastcall sub_1D7F160(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned int v4; // r12d
  __int64 v6; // rsi
  unsigned __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v10; // r14
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  int v13; // [rsp+8h] [rbp-28h]

  v4 = a3;
  v6 = *(_QWORD *)(a2 + 32);
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((a4 - v6) >> 3);
  switch ( **(_WORD **)(a2 + 16) )
  {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xF:
      return v4;
    case 7:
      v12 = *(_QWORD *)(v6 + 104);
      if ( !(_DWORD)v12 )
        return v4;
      return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, unsigned __int64))(**(_QWORD **)(a1 + 240) + 128LL))(
               *(_QWORD *)(a1 + 240),
               v12,
               a3,
               v7);
    case 8:
      v10 = *(_QWORD *)(v6 + 144);
      result = a3;
      if ( (_DWORD)v10 )
      {
        v13 = v7;
        result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 240) + 136LL))(
                   *(_QWORD *)(a1 + 240),
                   (unsigned int)v10,
                   a3);
        if ( v13 == 2 )
          return result;
      }
      else if ( (_DWORD)v7 == 2 )
      {
        return result;
      }
      v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 24LL)
                      + 16LL * (*(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL) & 0x7FFFFFFF))
          & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_BYTE *)(v11 + 30) )
        v4 &= ~*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 248LL) + 4LL * (unsigned int)v10);
      else
        return *(unsigned int *)(v11 + 24);
      return v4;
    case 0xE:
      v8 = *(_QWORD *)(v6 + 40LL * (unsigned int)(v7 + 1) + 24);
      if ( (_DWORD)v8 )
        return (*(unsigned int (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 240) + 136LL))(
                 *(_QWORD *)(a1 + 240),
                 v8,
                 a3);
      return v4;
  }
}
