// Function: sub_3982B10
// Address: 0x3982b10
//
void __fastcall sub_3982B10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  switch ( *(_DWORD *)a1 )
  {
    case 1:
      sub_39820D0((__int64 *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
      break;
    case 2:
      sub_3982460((__int64 *)(a1 + 8), a2, *(_WORD *)(a1 + 6), a4, a5, a6);
      break;
    case 3:
      sub_3982290((_QWORD *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
      break;
    case 4:
      sub_3982300((__int64 *)(a1 + 8), a2, *(_WORD *)(a1 + 6), a4, a5, a6);
      break;
    case 5:
      sub_3982430(*(_QWORD *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
      break;
    case 6:
      sub_39826E0((__int64 *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
      break;
    case 7:
      sub_39830D0(*(_QWORD *)(a1 + 8), a2, *(unsigned __int16 *)(a1 + 6));
      break;
    case 8:
      sub_3982F90(*(_QWORD *)(a1 + 8), a2, *(unsigned __int16 *)(a1 + 6));
      break;
    case 9:
      sub_397C410(
        a2,
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 504) + 1192LL) + 32LL * *(_QWORD *)(a1 + 8) + 8),
        *(_BYTE *)(*(_QWORD *)(a2 + 504) + 4513LL));
      break;
    case 0xA:
      sub_3982560(*(_QWORD **)(a1 + 8), a2);
      break;
    default:
      return;
  }
}
