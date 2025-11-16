// Function: sub_160EEE0
// Address: 0x160eee0
//
void __fastcall sub_160EEE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  unsigned int v8; // r12d
  unsigned int v9; // ebx
  unsigned int v10; // r13d
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // esi
  __int64 *v15; // rdx
  __int64 v16; // r9
  __int64 v17; // rax
  int v18; // edx
  int v19; // r10d

  v4 = sub_16BA580(a1, a2, a3);
  v5 = sub_16E8750(v4, (unsigned int)(2 * a2));
  v6 = *(__m128i **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0x12u )
  {
    sub_16E7EE0(v5, "ModulePass Manager\n", 19);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F552A0);
    v6[1].m128i_i8[2] = 10;
    v6[1].m128i_i16[0] = 29285;
    *v6 = si128;
    *(_QWORD *)(v5 + 24) += 19LL;
  }
  v8 = a2 + 1;
  v9 = 0;
  v10 = a2 + 2;
  while ( *(_DWORD *)(a1 + 192) > v9 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * v9);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v11 + 136LL))(v11, v8);
    v12 = *(unsigned int *)(a1 + 592);
    if ( (_DWORD)v12 )
    {
      v13 = *(_QWORD *)(a1 + 576);
      v14 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( v11 == *v15 )
      {
LABEL_6:
        if ( v15 != (__int64 *)(v13 + 16 * v12) )
        {
          v17 = *(_QWORD *)(a1 + 600) + 16LL * *((unsigned int *)v15 + 2);
          if ( *(_QWORD *)(a1 + 608) != v17 )
            (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(v17 + 8) + 136LL))(*(_QWORD *)(v17 + 8), v10);
        }
      }
      else
      {
        v18 = 1;
        while ( v16 != -8 )
        {
          v19 = v18 + 1;
          v14 = (v12 - 1) & (v18 + v14);
          v15 = (__int64 *)(v13 + 16LL * v14);
          v16 = *v15;
          if ( v11 == *v15 )
            goto LABEL_6;
          v18 = v19;
        }
      }
    }
    ++v9;
    sub_160EBB0(a1 + 160, v11, v8);
  }
}
