// Function: sub_C6A4F0
// Address: 0xc6a4f0
//
unsigned __int16 *__fastcall sub_C6A4F0(__int64 a1, unsigned __int16 *a2)
{
  unsigned __int16 *result; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int16 *v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax

  result = (unsigned __int16 *)*a2;
  *(_WORD *)a1 = (_WORD)result;
  switch ( (__int16)result )
  {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
      *(__m128i *)(a1 + 8) = _mm_loadu_si128((const __m128i *)(a2 + 4));
      *(__m128i *)(a1 + 24) = _mm_loadu_si128((const __m128i *)(a2 + 12));
      break;
    case 5:
      *(__m128i *)(a1 + 8) = _mm_loadu_si128((const __m128i *)(a2 + 4));
      break;
    case 6:
      *(_QWORD *)(a1 + 8) = a1 + 24;
      v6 = (unsigned __int16 *)*((_QWORD *)a2 + 1);
      result = a2 + 12;
      if ( a2 + 12 == v6 )
      {
        *(__m128i *)(a1 + 24) = _mm_loadu_si128((const __m128i *)(a2 + 12));
      }
      else
      {
        *(_QWORD *)(a1 + 8) = v6;
        *(_QWORD *)(a1 + 24) = *((_QWORD *)a2 + 3);
      }
      v7 = *((_QWORD *)a2 + 2);
      *((_QWORD *)a2 + 1) = result;
      *((_QWORD *)a2 + 2) = 0;
      *(_QWORD *)(a1 + 16) = v7;
      *((_BYTE *)a2 + 24) = 0;
      *a2 = 0;
      break;
    case 7:
      *(_QWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 8) = 1;
      v8 = *((_QWORD *)a2 + 2);
      ++*((_QWORD *)a2 + 1);
      v9 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v8;
      LODWORD(v8) = *((_DWORD *)a2 + 6);
      *((_QWORD *)a2 + 2) = v9;
      LODWORD(v9) = *(_DWORD *)(a1 + 24);
      *(_DWORD *)(a1 + 24) = v8;
      LODWORD(v8) = *((_DWORD *)a2 + 7);
      *((_DWORD *)a2 + 6) = v9;
      LODWORD(v9) = *(_DWORD *)(a1 + 28);
      *(_DWORD *)(a1 + 28) = v8;
      LODWORD(v8) = *((_DWORD *)a2 + 8);
      *((_DWORD *)a2 + 7) = v9;
      result = (unsigned __int16 *)*(unsigned int *)(a1 + 32);
      *(_DWORD *)(a1 + 32) = v8;
      *((_DWORD *)a2 + 8) = (_DWORD)result;
      *a2 = 0;
      break;
    case 8:
      v3 = *((_QWORD *)a2 + 1);
      *((_QWORD *)a2 + 1) = 0;
      *(_QWORD *)(a1 + 8) = v3;
      v4 = *((_QWORD *)a2 + 2);
      *((_QWORD *)a2 + 2) = 0;
      *(_QWORD *)(a1 + 16) = v4;
      v5 = *((_QWORD *)a2 + 3);
      *((_QWORD *)a2 + 3) = 0;
      *(_QWORD *)(a1 + 24) = v5;
      *a2 = 0;
      result = 0;
      break;
    default:
      return result;
  }
  return result;
}
