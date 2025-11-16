// Function: sub_6CFD10
// Address: 0x6cfd10
//
char __fastcall sub_6CFD10(unsigned __int16 a1, __int64 *a2, __int64 a3, __m128i *a4, __int64 *a5)
{
  __int16 v7; // bx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  char result; // al
  _DWORD v14[9]; // [rsp+Ch] [rbp-24h] BYREF

  v7 = a1 - 33;
  v8 = (__int64)*(&off_4B6DFA0 + a1);
  sub_7CB300(v8, 0, 0, 1, a3);
  switch ( v7 )
  {
    case 0:
    case 17:
    case 18:
      result = sub_6B38B0(a2, 0, (__int64)a5, v10);
      break;
    case 1:
    case 6:
    case 7:
      result = sub_6B1D00(a2, 0, (__int64)a5, v10);
      break;
    case 2:
    case 3:
      result = sub_6B21E0(a2, 0, (__int64)a5, v10);
      break;
    case 8:
    case 9:
      result = sub_6B2B40(a2, 0, (__int64)a5, v10);
      break;
    case 10:
    case 11:
    case 12:
    case 13:
      result = sub_6B2F50(a2, 0, (__int64)a5, v10);
      break;
    case 14:
    case 15:
      result = sub_6B3030(a2, 0, a5, v10);
      break;
    case 19:
    case 20:
      result = sub_6B3BD0(a2, 0, 0, (__int64)a5);
      break;
    case 23:
      result = sub_6CEC90((__int64)a2, 0, v14, a5, v11);
      break;
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 30:
    case 31:
    case 32:
    case 33:
      result = sub_6CF140((__int64)a2, 0, v14, a5, v11);
      break;
    case 34:
      result = sub_6B70D0(a2, 0, (__int64)a5);
      break;
    case 114:
    case 115:
      result = sub_6B0A80((__int64)a2, 0, 0, (__int64)a5, a4, v12);
      break;
    default:
      sub_7B8B50(v8, 0, v9, v10);
      sub_6E6800(a5);
      result = sub_6E6260(a5);
      break;
  }
  return result;
}
