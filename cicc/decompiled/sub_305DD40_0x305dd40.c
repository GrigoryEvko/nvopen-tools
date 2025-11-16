// Function: sub_305DD40
// Address: 0x305dd40
//
__int64 __fastcall sub_305DD40(__int64 a1, _QWORD *a2, unsigned int a3, unsigned int a4, unsigned int a5, __int64 a6)
{
  unsigned int v6; // eax
  __int64 v10; // rdx
  __int64 v11; // r11
  __int16 v12; // dx
  __int64 v13; // rdi
  __int64 (*v14)(); // r10

  switch ( a3 )
  {
    case 1u:
      v12 = 2;
      break;
    case 2u:
      v12 = 3;
      break;
    case 4u:
      v12 = 4;
      break;
    case 8u:
      v12 = 5;
      break;
    case 0x10u:
      v12 = 6;
      break;
    case 0x20u:
      v12 = 7;
      break;
    case 0x40u:
      v12 = 8;
      break;
    case 0x80u:
      v12 = 9;
      break;
    default:
      v6 = sub_3007020(a2, a3);
      v11 = v10;
      v12 = v6;
      goto LABEL_12;
  }
  v11 = 0;
LABEL_12:
  v13 = *(_QWORD *)(a1 + 32);
  LOWORD(v6) = v12;
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 808LL);
  if ( v14 == sub_2D56600 )
    return 0;
  else
    return ((unsigned int (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, __int64))v14)(
             v13,
             v6,
             v11,
             a4,
             a5,
             0,
             a6);
}
