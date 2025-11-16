// Function: sub_2A35880
// Address: 0x2a35880
//
_QWORD *__fastcall sub_2A35880(_QWORD *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  __m128i *v9; // r8
  __m128i *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 *v13; // rsi
  int v14; // r11d
  unsigned int i; // eax
  unsigned int v16; // eax
  __int64 v17; // rdx
  char v18; // al
  _QWORD *v19; // rsi
  _QWORD *v20; // rdx

  v8 = sub_BC1CD0(a4, &unk_4FDBCC8, a3);
  v11 = *(unsigned int *)(a4 + 88);
  v12 = *(_QWORD *)(a4 + 72);
  v13 = (__int64 *)(v8 + 8);
  if ( !(_DWORD)v11 )
    goto LABEL_14;
  v14 = 1;
  v10 = (__m128i *)(unsigned int)(v11 - 1);
  for ( i = (unsigned int)v10
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (unsigned int)v10 & v16 )
  {
    v9 = (__m128i *)(v12 + 24LL * i);
    if ( (_UNKNOWN *)v9->m128i_i64[0] == &unk_4F86630 && a3 == v9->m128i_i64[1] )
      break;
    if ( v9->m128i_i64[0] == -4096 && v9->m128i_i64[1] == -4096 )
      goto LABEL_14;
    v16 = v14 + i;
    ++v14;
  }
  if ( v9 == (__m128i *)(v12 + 24 * v11) )
  {
LABEL_14:
    v17 = 0;
  }
  else
  {
    v17 = *(_QWORD *)(v9[1].m128i_i64[0] + 24);
    if ( v17 )
      v17 += 8;
  }
  v18 = sub_2A33580(a3, v13, v17, *a2, v9, v10);
  v19 = a1 + 4;
  v20 = a1 + 10;
  if ( v18 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v19;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v20;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v19;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v20;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
