// Function: sub_16F3DE0
// Address: 0x16f3de0
//
void __fastcall sub_16F3DE0(__int64 a1, char *a2, __int64 a3)
{
  char v3; // al
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 *v7; // r8
  unsigned __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 i; // r12

  v3 = *a2;
  *(_BYTE *)a1 = *a2;
  switch ( v3 )
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
      sub_16F1520((__int64 *)(a1 + 8), *((_BYTE **)a2 + 1), *((_QWORD *)a2 + 1) + *((_QWORD *)a2 + 2));
      break;
    case 6:
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      sub_16DB620(a1 + 8);
      j___libc_free_0(*(_QWORD *)(a1 + 16));
      v4 = *((_DWORD *)a2 + 8);
      *(_DWORD *)(a1 + 32) = v4;
      if ( v4 )
      {
        *(_QWORD *)(a1 + 16) = sub_22077B0((unsigned __int64)v4 << 6);
        sub_16F39A0(a1 + 8, (__int64)(a2 + 8), v5, v6, v7);
      }
      else
      {
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
      }
      break;
    case 7:
      v8 = *((_QWORD *)a2 + 2) - *((_QWORD *)a2 + 1);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      if ( v8 )
      {
        if ( v8 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(a1, a2, a3);
        v9 = sub_22077B0(v8);
      }
      else
      {
        v8 = 0;
        v9 = 0;
      }
      *(_QWORD *)(a1 + 8) = v9;
      *(_QWORD *)(a1 + 16) = v9;
      *(_QWORD *)(a1 + 24) = v9 + v8;
      v10 = *((_QWORD *)a2 + 2);
      for ( i = *((_QWORD *)a2 + 1); v10 != i; v9 += 40 )
      {
        if ( v9 )
          sub_16F3DE0(v9, i);
        i += 40;
      }
      *(_QWORD *)(a1 + 16) = v9;
      break;
    default:
      return;
  }
}
