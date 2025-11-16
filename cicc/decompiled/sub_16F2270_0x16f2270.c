// Function: sub_16F2270
// Address: 0x16f2270
//
unsigned __int8 *__fastcall sub_16F2270(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 *result; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int8 *v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rax

  result = (unsigned __int8 *)*a2;
  *(_BYTE *)a1 = (_BYTE)result;
  switch ( (char)result )
  {
    case 0:
    case 1:
    case 2:
    case 3:
      *(__m128i *)(a1 + 8) = _mm_loadu_si128((const __m128i *)(a2 + 8));
      *(__m128i *)(a1 + 24) = _mm_loadu_si128((const __m128i *)(a2 + 24));
      break;
    case 4:
      *(__m128i *)(a1 + 8) = _mm_loadu_si128((const __m128i *)(a2 + 8));
      break;
    case 5:
      *(_QWORD *)(a1 + 8) = a1 + 24;
      v5 = (unsigned __int8 *)*((_QWORD *)a2 + 1);
      result = a2 + 24;
      if ( a2 + 24 == v5 )
      {
        *(__m128i *)(a1 + 24) = _mm_loadu_si128((const __m128i *)(a2 + 24));
      }
      else
      {
        *(_QWORD *)(a1 + 8) = v5;
        *(_QWORD *)(a1 + 24) = *((_QWORD *)a2 + 3);
      }
      v6 = *((_QWORD *)a2 + 2);
      *((_QWORD *)a2 + 1) = result;
      *((_QWORD *)a2 + 2) = 0;
      *(_QWORD *)(a1 + 16) = v6;
      a2[24] = 0;
      *a2 = 0;
      break;
    case 6:
      *(_QWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 8) = 1;
      v7 = *((_QWORD *)a2 + 2);
      ++*((_QWORD *)a2 + 1);
      v8 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v7;
      LODWORD(v7) = *((_DWORD *)a2 + 6);
      *((_QWORD *)a2 + 2) = v8;
      LODWORD(v8) = *(_DWORD *)(a1 + 24);
      *(_DWORD *)(a1 + 24) = v7;
      LODWORD(v7) = *((_DWORD *)a2 + 7);
      *((_DWORD *)a2 + 6) = v8;
      LODWORD(v8) = *(_DWORD *)(a1 + 28);
      *(_DWORD *)(a1 + 28) = v7;
      LODWORD(v7) = *((_DWORD *)a2 + 8);
      *((_DWORD *)a2 + 7) = v8;
      result = (unsigned __int8 *)*(unsigned int *)(a1 + 32);
      *(_DWORD *)(a1 + 32) = v7;
      *((_DWORD *)a2 + 8) = (_DWORD)result;
      *a2 = 0;
      break;
    case 7:
      v3 = *((_QWORD *)a2 + 1);
      *((_QWORD *)a2 + 1) = 0;
      *(_QWORD *)(a1 + 8) = v3;
      v4 = *((_QWORD *)a2 + 2);
      *((_QWORD *)a2 + 2) = 0;
      *(_QWORD *)(a1 + 16) = v4;
      result = (unsigned __int8 *)*((_QWORD *)a2 + 3);
      *((_QWORD *)a2 + 3) = 0;
      *(_QWORD *)(a1 + 24) = result;
      *a2 = 0;
      break;
    default:
      return result;
  }
  return result;
}
