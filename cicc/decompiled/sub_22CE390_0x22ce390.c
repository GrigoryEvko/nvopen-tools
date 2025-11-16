// Function: sub_22CE390
// Address: 0x22ce390
//
void __fastcall sub_22CE390(__int64 a1, unsigned __int8 *a2, unsigned __int8 **a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  unsigned __int8 **v8; // rax
  char v9; // dl
  __int64 v10; // r13
  __m128i *v11; // rdx
  __m128i si128; // xmm0
  void *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  _BYTE *v17; // rax
  unsigned __int8 v18[80]; // [rsp+0h] [rbp-50h] BYREF

  v7 = *(_QWORD *)a1;
  if ( !*(_BYTE *)(v7 + 28) )
    goto LABEL_8;
  v8 = *(unsigned __int8 ***)(v7 + 8);
  a4 = *(unsigned int *)(v7 + 20);
  a3 = &v8[a4];
  if ( v8 == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(v7 + 16) )
    {
      *(_DWORD *)(v7 + 20) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)v7;
LABEL_9:
      sub_22CDEF0((__int64)v18, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL), **(_QWORD **)(a1 + 16), (__int64)a2, 0);
      v10 = *(_QWORD *)(a1 + 24);
      v11 = *(__m128i **)(v10 + 32);
      if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 0x12u )
      {
        v10 = sub_CB6200(*(_QWORD *)(a1 + 24), "; LatticeVal for: '", 0x13u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_428A450);
        v11[1].m128i_i8[2] = 39;
        v11[1].m128i_i16[0] = 8250;
        *v11 = si128;
        *(_QWORD *)(v10 + 32) += 19LL;
      }
      sub_A69870(**(_QWORD **)(a1 + 16), (_BYTE *)v10, 0);
      v13 = *(void **)(v10 + 32);
      if ( *(_QWORD *)(v10 + 24) - (_QWORD)v13 <= 9u )
      {
        sub_CB6200(v10, "' in BB: '", 0xAu);
      }
      else
      {
        qmemcpy(v13, "' in BB: '", 10);
        *(_QWORD *)(v10 + 32) += 10LL;
      }
      sub_A5BF40(a2, *(_QWORD *)(a1 + 24), 0, 0);
      v14 = *(_QWORD *)(a1 + 24);
      v15 = *(_QWORD *)(v14 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v14 + 24) - v15) <= 5 )
      {
        v14 = sub_CB6200(v14, "' is: ", 6u);
      }
      else
      {
        *(_DWORD *)v15 = 1936269351;
        *(_WORD *)(v15 + 4) = 8250;
        *(_QWORD *)(v14 + 32) += 6LL;
      }
      v16 = sub_22EAFB0(v14, v18);
      v17 = *(_BYTE **)(v16 + 32);
      if ( *(_BYTE **)(v16 + 24) == v17 )
      {
        sub_CB6200(v16, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v17 = 10;
        ++*(_QWORD *)(v16 + 32);
      }
      sub_22C0090(v18);
      return;
    }
LABEL_8:
    sub_C8CC70(v7, (__int64)a2, (__int64)a3, a4, a5, a6);
    if ( !v9 )
      return;
    goto LABEL_9;
  }
  while ( a2 != *v8 )
  {
    if ( a3 == ++v8 )
      goto LABEL_7;
  }
}
