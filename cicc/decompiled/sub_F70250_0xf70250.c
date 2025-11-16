// Function: sub_F70250
// Address: 0xf70250
//
__int64 __fastcall sub_F70250(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rsi
  __int64 result; // rax
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // rax
  unsigned int v10; // [rsp+8h] [rbp-48h]
  _BYTE v11[32]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v12; // [rsp+30h] [rbp-20h]

  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 8) + 24LL);
  switch ( a3 )
  {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 12:
    case 13:
    case 14:
    case 15:
      v12 = 257;
      result = sub_B33BC0(a1, dword_3F8AE00[a3 - 1], a2, v10, (__int64)v11);
      break;
    case 10:
    case 16:
      v8 = sub_F70230(a3, v6, *(unsigned int *)(a1 + 104), a4, a5);
      result = sub_B348A0(a1, (__int64)v8, a2);
      break;
    case 11:
      v9 = sub_F70230(11, v6, *(unsigned int *)(a1 + 104), a4, a5);
      result = sub_B348F0(a1, (__int64)v9, a2);
      break;
    default:
      BUG();
  }
  return result;
}
