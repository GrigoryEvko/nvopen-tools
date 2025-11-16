// Function: sub_C6CEC0
// Address: 0xc6cec0
//
void __fastcall sub_C6CEC0(__int64 a1, __int16 *a2, __int64 a3)
{
  __int16 v3; // ax
  unsigned int v4; // eax
  unsigned __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 i; // r12

  v3 = *a2;
  *(_WORD *)a1 = *a2;
  switch ( v3 )
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
      sub_C68E20((__int64 *)(a1 + 8), *((_BYTE **)a2 + 1), *((_QWORD *)a2 + 1) + *((_QWORD *)a2 + 2));
      break;
    case 7:
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      sub_C6B900(a1 + 8);
      sub_C7D6A0(*(_QWORD *)(a1 + 16), (unsigned __int64)*(unsigned int *)(a1 + 32) << 6, 8);
      v4 = *((_DWORD *)a2 + 8);
      *(_DWORD *)(a1 + 32) = v4;
      if ( v4 )
      {
        *(_QWORD *)(a1 + 16) = sub_C7D670((unsigned __int64)v4 << 6, 8);
        sub_C6CAA0(a1 + 8, (__int64)(a2 + 4));
      }
      else
      {
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
      }
      break;
    case 8:
      v5 = *((_QWORD *)a2 + 2) - *((_QWORD *)a2 + 1);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      if ( v5 )
      {
        if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(a1, a2, a3);
        v6 = sub_22077B0(v5);
      }
      else
      {
        v5 = 0;
        v6 = 0;
      }
      *(_QWORD *)(a1 + 8) = v6;
      *(_QWORD *)(a1 + 16) = v6;
      *(_QWORD *)(a1 + 24) = v6 + v5;
      v7 = *((_QWORD *)a2 + 2);
      for ( i = *((_QWORD *)a2 + 1); v7 != i; v6 += 40 )
      {
        if ( v6 )
          sub_C6CEC0(v6, i);
        i += 40;
      }
      *(_QWORD *)(a1 + 16) = v6;
      break;
    default:
      return;
  }
}
