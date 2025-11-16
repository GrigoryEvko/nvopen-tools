// Function: sub_1D7F2C0
// Address: 0x1d7f2c0
//
__int64 __fastcall sub_1D7F2C0(__int64 a1, __int64 a2, int a3, unsigned int a4)
{
  __int64 v6; // rcx
  _QWORD *v8; // rdi
  __int64 v9; // r14
  unsigned int v10; // eax
  _QWORD *v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned int v15; // eax

  v6 = *(_QWORD *)(a2 + 16);
  switch ( **(_WORD **)(v6 + 16) )
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
      return a4 & (unsigned int)sub_1E69F40(*(_QWORD *)(a1 + 232), *(unsigned int *)(a2 + 8));
    case 7:
      v14 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 104LL);
      if ( (_DWORD)v14 )
        a4 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 240) + 136LL))(
               *(_QWORD *)(a1 + 240),
               v14,
               a4);
      break;
    case 8:
      v12 = *(_QWORD **)(a1 + 240);
      v13 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 144LL);
      if ( a3 == 2 )
      {
        if ( (_DWORD)v13 )
        {
          v15 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(*v12 + 128LL))(*(_QWORD *)(a1 + 240), v13, a4);
          v12 = *(_QWORD **)(a1 + 240);
          a4 = v15;
        }
        a4 &= *(_DWORD *)(v12[31] + 4LL * (unsigned int)v13);
      }
      else
      {
        a4 &= ~*(_DWORD *)(v12[31] + 4LL * (unsigned int)v13);
      }
      break;
    case 0xE:
      v8 = *(_QWORD **)(a1 + 240);
      v9 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 40LL * (unsigned int)(a3 + 1) + 24);
      if ( (_DWORD)v9 )
      {
        v10 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*v8 + 128LL))(v8, (unsigned int)v9, a4);
        v8 = *(_QWORD **)(a1 + 240);
        a4 = v10;
      }
      a4 &= *(_DWORD *)(v8[31] + 4LL * (unsigned int)v9);
      break;
  }
  return a4 & (unsigned int)sub_1E69F40(*(_QWORD *)(a1 + 232), *(unsigned int *)(a2 + 8));
}
