// Function: sub_A51070
// Address: 0xa51070
//
char *__fastcall sub_A51070(char *a1, int a2)
{
  __int64 v2; // rbp
  __int64 v3; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD v10[4]; // [rsp-20h] [rbp-20h] BYREF

  v10[3] = v2;
  v10[2] = v3;
  switch ( a2 )
  {
    case 0:
      *((_QWORD *)a1 + 1) = 8;
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "external");
      break;
    case 1:
      v10[0] = 20;
      *(_QWORD *)a1 = a1 + 16;
      v5 = sub_22409D0(a1, v10, 0);
      v6 = v10[0];
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F24A90);
      *(_QWORD *)a1 = v5;
      *((_QWORD *)a1 + 2) = v6;
      *(_DWORD *)(v5 + 16) = 2037148769;
      *(__m128i *)v5 = si128;
      v8 = v10[0];
      v9 = *(_QWORD *)a1;
      *((_QWORD *)a1 + 1) = v10[0];
      *(_BYTE *)(v9 + v8) = 0;
      break;
    case 2:
      sub_A4F810((__int64 *)a1, "linkonce");
      break;
    case 3:
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "linkonce_odr");
      *((_QWORD *)a1 + 1) = 12;
      break;
    case 4:
      sub_A4F810((__int64 *)a1, "weak");
      break;
    case 5:
      sub_A4F810((__int64 *)a1, "weak_odr");
      break;
    case 6:
      sub_A4F810((__int64 *)a1, "appending");
      break;
    case 7:
      sub_A4F810((__int64 *)a1, "internal");
      break;
    case 8:
      sub_A4F810((__int64 *)a1, "private");
      break;
    case 9:
      sub_A4F810((__int64 *)a1, "extern_weak");
      break;
    case 10:
      sub_A4F810((__int64 *)a1, "common");
      break;
    default:
      BUG();
  }
  return a1;
}
