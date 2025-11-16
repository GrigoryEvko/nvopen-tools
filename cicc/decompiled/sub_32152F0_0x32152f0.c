// Function: sub_32152F0
// Address: 0x32152f0
//
__int64 (*__fastcall sub_32152F0(__int64 *a1, _QWORD **a2, unsigned __int16 a3))(void)
{
  __int64 (*result)(void); // rax
  _QWORD *v5; // r13
  __int64 (__fastcall *v6)(_QWORD *, __int64, _QWORD); // r14
  unsigned __int64 v7; // rax
  unsigned int v8; // eax
  unsigned int v9; // [rsp+Ah] [rbp-26h] BYREF
  __int16 v10; // [rsp+Eh] [rbp-22h]

  if ( a3 <= 0x2Cu )
  {
    if ( a3 )
    {
      switch ( a3 )
      {
        case 1u:
        case 5u:
        case 6u:
        case 7u:
        case 0xBu:
        case 0xCu:
        case 0xEu:
        case 0x10u:
        case 0x11u:
        case 0x12u:
        case 0x13u:
        case 0x14u:
        case 0x17u:
        case 0x1Cu:
        case 0x1Du:
        case 0x1Fu:
        case 0x20u:
        case 0x24u:
        case 0x25u:
        case 0x26u:
        case 0x27u:
        case 0x28u:
        case 0x29u:
        case 0x2Au:
        case 0x2Bu:
        case 0x2Cu:
          goto LABEL_8;
        case 0xDu:
          return (__int64 (*)(void))((__int64 (__fastcall *)(_QWORD **, __int64, _QWORD))(*a2)[52])(a2, *a1, 0);
        case 0xFu:
        case 0x15u:
        case 0x1Au:
        case 0x1Bu:
        case 0x23u:
          return (__int64 (*)(void))((__int64 (__fastcall *)(_QWORD **, __int64, _QWORD, _QWORD))(*a2)[53])(
                                      a2,
                                      *a1,
                                      0,
                                      0);
        case 0x19u:
        case 0x21u:
          result = *(__int64 (**)(void))(*a2[28] + 160LL);
          if ( (char *)result != (char *)nullsub_99 )
            return (__int64 (*)(void))result();
          return result;
        default:
          break;
      }
    }
LABEL_13:
    BUG();
  }
  if ( a3 > 0x1F02u )
  {
    if ( (unsigned __int16)(a3 - 7968) > 1u )
      goto LABEL_13;
LABEL_8:
    v5 = a2[28];
    v6 = *(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*v5 + 536LL);
    v7 = sub_31DF6E0((__int64)a2);
    v9 = v7;
    v10 = WORD2(v7);
    v8 = sub_3215240(a1, &v9, a3);
    return (__int64 (*)(void))v6(v5, *a1, v8);
  }
  else
  {
    if ( a3 <= 0x1F00u )
      goto LABEL_13;
    return (__int64 (*)(void))((__int64 (__fastcall *)(_QWORD **, __int64, _QWORD, _QWORD))(*a2)[53])(a2, *a1, 0, 0);
  }
}
