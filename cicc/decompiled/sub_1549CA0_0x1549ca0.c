// Function: sub_1549CA0
// Address: 0x1549ca0
//
char *__fastcall sub_1549CA0(char *a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // [rsp+8h] [rbp-18h] BYREF

  switch ( a2 )
  {
    case 0:
      *((_QWORD *)a1 + 1) = 8;
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "external");
      break;
    case 1:
      v8 = 20;
      *(_QWORD *)a1 = a1 + 16;
      v3 = sub_22409D0(a1, &v8, 0);
      v4 = v8;
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F24A90);
      *(_QWORD *)a1 = v3;
      *((_QWORD *)a1 + 2) = v4;
      *(_DWORD *)(v3 + 16) = 2037148769;
      *(__m128i *)v3 = si128;
      v6 = v8;
      v7 = *(_QWORD *)a1;
      *((_QWORD *)a1 + 1) = v8;
      *(_BYTE *)(v7 + v6) = 0;
      break;
    case 2:
      *((_QWORD *)a1 + 1) = 8;
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "linkonce");
      break;
    case 3:
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "linkonce_odr");
      *((_QWORD *)a1 + 1) = 12;
      break;
    case 4:
      sub_1548B10((__int64 *)a1, "weak");
      break;
    case 5:
      *((_QWORD *)a1 + 1) = 8;
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "weak_odr");
      break;
    case 6:
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "appending");
      *((_QWORD *)a1 + 1) = 9;
      break;
    case 7:
      sub_1548B10((__int64 *)a1, "internal");
      break;
    case 8:
      a1[22] = 101;
      *(_QWORD *)a1 = a1 + 16;
      *((_DWORD *)a1 + 4) = 1986622064;
      *((_WORD *)a1 + 10) = 29793;
      *((_QWORD *)a1 + 1) = 7;
      a1[23] = 0;
      break;
    case 9:
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "extern_weak");
      *((_QWORD *)a1 + 1) = 11;
      break;
    case 10:
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "common");
      *((_QWORD *)a1 + 1) = 6;
      break;
  }
  return a1;
}
