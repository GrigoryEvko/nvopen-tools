// Function: sub_1382000
// Address: 0x1382000
//
char __fastcall sub_1382000(unsigned int *a1)
{
  unsigned int v1; // edx
  unsigned int v2; // esi
  unsigned int v3; // r10d
  __int64 v4; // r13
  unsigned int v5; // ebx
  unsigned int v6; // eax
  unsigned int v7; // ecx
  unsigned int v8; // r9d
  unsigned int v9; // r11d
  __m128i v10; // xmm0
  __int64 v11; // rax

  v1 = *a1;
  v2 = a1[1];
  v3 = a1[2];
  v4 = *((_QWORD *)a1 + 2);
  v5 = a1[3];
  while ( 1 )
  {
    v6 = *(a1 - 6);
    if ( v6 <= v1 )
    {
      v7 = *(a1 - 5);
      if ( v7 <= v2 || v6 != v1 )
      {
        v8 = *(a1 - 4);
        v9 = *(a1 - 3);
        if ( v6 < v1 || v7 < v2 && v6 == v1 )
          break;
        if ( v8 <= v3 )
        {
          LOBYTE(v6) = v8 == v3;
          if ( (v9 <= v5 || v8 != v3) && (v8 < v3 || v9 < v5 && v8 == v3 || *((_QWORD *)a1 - 1) <= v4) )
            break;
        }
      }
    }
    v10 = _mm_loadu_si128((const __m128i *)(a1 - 6));
    v11 = *((_QWORD *)a1 - 1);
    a1 -= 6;
    *(__m128i *)(a1 + 6) = v10;
    *((_QWORD *)a1 + 5) = v11;
  }
  a1[1] = v2;
  a1[2] = v3;
  a1[3] = v5;
  *((_QWORD *)a1 + 2) = v4;
  *a1 = v1;
  return v6;
}
