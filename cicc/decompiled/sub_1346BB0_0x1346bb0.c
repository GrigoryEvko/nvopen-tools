// Function: sub_1346BB0
// Address: 0x1346bb0
//
_QWORD *__fastcall sub_1346BB0(__int64 a1, const __m128i *a2)
{
  char v2; // r14
  __int64 v5; // r8
  _QWORD *v6; // rsi
  __int64 v7; // r9
  _QWORD *v8; // r13
  __int64 *v9; // rdx
  _BYTE *v10; // rax
  __int64 v11; // rcx
  __m128i v13; // xmm0
  char *v14; // rdx
  __int64 v15; // rcx
  _BYTE *v16; // rax
  __int64 v17; // rdi
  __m128i v18; // [rsp+0h] [rbp-80h]
  __int64 v19; // [rsp+20h] [rbp-60h]
  _OWORD v20[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v21; // [rsp+50h] [rbp-30h]
  _BYTE v22[40]; // [rsp+58h] [rbp-28h] BYREF

  if ( pthread_mutex_trylock(&stru_4F96CA0) )
  {
    sub_130AD90((__int64)&unk_4F96C60);
    byte_4F96CC8 = 1;
  }
  ++qword_4F96C98;
  if ( a1 != qword_4F96C90 )
  {
    ++qword_4F96C88;
    qword_4F96C90 = a1;
  }
  v5 = 0;
  v6 = &unk_4F96CE0;
  while ( 1 )
  {
    v7 = *v6;
    v8 = v6;
    if ( (*v6 & 1) == 0 )
    {
      v9 = v6 + 1;
      v10 = v20;
      do
      {
        v11 = *v9;
        v10 += 8;
        ++v9;
        *((_QWORD *)v10 - 1) = v11;
      }
      while ( v10 != v22 );
      if ( v7 == *v6 )
        break;
    }
    if ( !v2 )
      goto LABEL_15;
LABEL_11:
    ++v5;
    v6 += 6;
    if ( v5 == 4 )
    {
      v8 = 0;
      goto LABEL_13;
    }
  }
  v2 = v21;
  v19 = v21;
  if ( (_BYTE)v21 )
    goto LABEL_11;
LABEL_15:
  v13 = _mm_loadu_si128(a2 + 1);
  LOBYTE(v19) = 1;
  v18 = _mm_loadu_si128(a2);
  v14 = (char *)&unk_4F96CE0 + 48 * v5 + 8;
  v21 = v19;
  v20[0] = v18;
  v20[1] = v13;
  v15 = (*v6)++;
  v16 = v20;
  do
  {
    v17 = *(_QWORD *)v16;
    v16 += 8;
    v14 += 8;
    *((_QWORD *)v14 - 1) = v17;
  }
  while ( v16 != v22 );
  *v6 = v15 + 2;
  ++dword_4F96DA0;
  sub_1313A10(a1);
LABEL_13:
  byte_4F96CC8 = 0;
  pthread_mutex_unlock(&stru_4F96CA0);
  return v8;
}
