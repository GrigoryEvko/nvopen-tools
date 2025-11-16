// Function: sub_35EDF20
// Address: 0x35edf20
//
__int64 __fastcall sub_35EDF20(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  unsigned __int64 v4; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 *v8; // [rsp+0h] [rbp-80h] BYREF
  __int16 v9; // [rsp+20h] [rbp-60h]
  unsigned __int64 v10[4]; // [rsp+30h] [rbp-50h] BYREF
  char v11; // [rsp+50h] [rbp-30h]
  void *v12; // [rsp+58h] [rbp-28h] BYREF
  int v13; // [rsp+60h] [rbp-20h]
  void **v14; // [rsp+68h] [rbp-18h] BYREF

  switch ( a2 )
  {
    case 0:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "NotAtomic");
      *(_QWORD *)(a1 + 8) = 9;
      result = a1;
      break;
    case 2:
      *(_BYTE *)(a1 + 22) = 100;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1634493778;
      *(_WORD *)(a1 + 20) = 25976;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      result = a1;
      break;
    case 4:
      *(_DWORD *)(a1 + 16) = 1970365249;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 20) = 29289;
      *(_BYTE *)(a1 + 22) = 101;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      result = a1;
      break;
    case 5:
      *(_BYTE *)(a1 + 22) = 101;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1701602642;
      *(_WORD *)(a1 + 20) = 29537;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      result = a1;
      break;
    case 6:
      strcpy((char *)(a1 + 16), "AcquireRelease");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 14;
      result = a1;
      break;
    case 7:
      v10[0] = 22;
      *(_QWORD *)a1 = a1 + 16;
      v3 = sub_22409D0(a1, v10, 0);
      v4 = v10[0];
      si128 = _mm_load_si128((const __m128i *)&xmmword_44FE7E0);
      *(_QWORD *)a1 = v3;
      *(_QWORD *)(a1 + 16) = v4;
      *(_DWORD *)(v3 + 16) = 1702130537;
      *(_WORD *)(v3 + 20) = 29806;
      *(__m128i *)v3 = si128;
      v6 = v10[0];
      v7 = *(_QWORD *)a1;
      *(_QWORD *)(a1 + 8) = v10[0];
      *(_BYTE *)(v7 + v6) = 0;
      result = a1;
      break;
    case 8:
      *(_QWORD *)(a1 + 8) = 8;
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "Volatile");
      result = a1;
      break;
    case 9:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "RelaxedMMIO");
      *(_QWORD *)(a1 + 8) = 11;
      result = a1;
      break;
    default:
      v13 = a2;
      v10[2] = (unsigned __int64)&v14;
      v10[0] = (unsigned __int64)"Unknown NVPTX::Ordering \"{}\".";
      v10[1] = 29;
      v12 = &unk_49E65E8;
      v10[3] = 1;
      v11 = 1;
      v14 = &v12;
      v9 = 263;
      v8 = v10;
      sub_C64D30((__int64)&v8, 1u);
  }
  return result;
}
