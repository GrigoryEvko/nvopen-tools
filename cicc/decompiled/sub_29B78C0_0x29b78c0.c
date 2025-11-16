// Function: sub_29B78C0
// Address: 0x29b78c0
//
__int64 *__fastcall sub_29B78C0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4, int a5)
{
  __int64 v6; // rcx
  __int64 *result; // rax
  __int64 v8; // r11
  __int64 v9; // r10
  __int64 v10; // rdi
  __int64 v11; // r9

  v6 = *a2;
  result = a1;
  v8 = a2[1];
  v9 = *a3;
  v10 = *a2 + 8 * a4;
  v11 = a3[1];
  switch ( a5 )
  {
    case 0:
      *result = v6;
      result[1] = v8;
      result[2] = v9;
      result[3] = v11;
      result[4] = qword_5007B10;
      result[5] = qword_5007B18;
      break;
    case 1:
      *result = v9;
      result[1] = v11;
      result[2] = v6;
      result[3] = v8;
      result[4] = qword_5007B10;
      result[5] = qword_5007B18;
      break;
    case 2:
      *result = v6;
      result[1] = v10;
      result[2] = v9;
      result[3] = v11;
      result[4] = v10;
      result[5] = v8;
      break;
    case 3:
      *result = v9;
      result[1] = v11;
      result[2] = v10;
      result[3] = v8;
      result[4] = v6;
      result[5] = v10;
      break;
    case 4:
      *result = v10;
      result[1] = v8;
      result[2] = v6;
      result[3] = v10;
      result[4] = v9;
      result[5] = v11;
      break;
    default:
      BUG();
  }
  return result;
}
