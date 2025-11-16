// Function: sub_1DC9360
// Address: 0x1dc9360
//
unsigned __int64 __fastcall sub_1DC9360(_QWORD *a1, __int64 a2)
{
  __m128i *v4; // rdx
  unsigned __int64 result; // rax
  __int64 *v6; // r14
  _QWORD *v7; // rbx
  __int64 v8; // rax
  int v9; // esi
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _WORD *v13; // rdx
  void *v14; // rcx
  unsigned __int64 v15; // rax
  __int64 v16; // r13
  char *v17; // rsi
  size_t v18; // rax
  char *v19; // rdi
  size_t v20; // rdx
  __int64 v21; // rax
  char *v22; // rax
  bool v23; // cf
  __int64 v24; // rax
  char *src; // [rsp+8h] [rbp-38h]
  void *srca; // [rsp+8h] [rbp-38h]

  v4 = *(__m128i **)(a2 + 24);
  result = *(_QWORD *)(a2 + 16) - (_QWORD)v4;
  if ( result <= 0x1F )
  {
    result = sub_16E7EE0(a2, "********** INTERVALS **********\n", 0x20u);
  }
  else
  {
    *v4 = _mm_load_si128((const __m128i *)&xmmword_42E9A40);
    v4[1] = _mm_load_si128((const __m128i *)&xmmword_42E9A50);
    *(_QWORD *)(a2 + 24) += 32LL;
  }
  v6 = (__int64 *)a1[45];
  v7 = a1 + 51;
  if ( v6 )
  {
    while ( 1 )
    {
      sub_1DB53F0((__int64)(v6 + 2), a2);
      v8 = a1[52];
      v9 = *((_DWORD *)v6 + 2);
      v10 = a1 + 51;
      if ( v8 )
      {
        do
        {
          while ( 1 )
          {
            v11 = *(_QWORD *)(v8 + 16);
            v12 = *(_QWORD *)(v8 + 24);
            if ( v9 <= *(_DWORD *)(v8 + 32) )
              break;
            v8 = *(_QWORD *)(v8 + 24);
            if ( !v12 )
              goto LABEL_9;
          }
          v10 = (_QWORD *)v8;
          v8 = *(_QWORD *)(v8 + 16);
        }
        while ( v11 );
LABEL_9:
        if ( v10 != v7 && v9 < *((_DWORD *)v10 + 8) )
          v10 = a1 + 51;
      }
      v13 = *(_WORD **)(a2 + 24);
      v14 = (void *)v10[5];
      v15 = *(_QWORD *)(a2 + 16) - (_QWORD)v13;
      if ( !v14 )
      {
        if ( v15 <= 0xA )
        {
          result = sub_16E7EE0(a2, " [Unknown]\n", 0xBu);
        }
        else
        {
          qmemcpy(v13, " [Unknown]\n", 11);
          result = 23918;
          *(_QWORD *)(a2 + 24) += 11LL;
        }
        goto LABEL_21;
      }
      if ( v15 <= 1 )
      {
        srca = (void *)v10[5];
        v24 = sub_16E7EE0(a2, " [", 2u);
        v14 = srca;
        v16 = v24;
      }
      else
      {
        v16 = a2;
        *v13 = 23328;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      v17 = (char *)(*(_QWORD *)(a1[29] + 80LL) + *(unsigned int *)(*(_QWORD *)v14 + 16LL));
      if ( !v17 )
        goto LABEL_24;
      v18 = strlen(v17);
      v19 = *(char **)(v16 + 24);
      v20 = v18;
      result = *(_QWORD *)(v16 + 16) - (_QWORD)v19;
      if ( v20 > result )
        break;
      if ( v20 )
      {
        src = (char *)v20;
        memcpy(v19, v17, v20);
        v21 = *(_QWORD *)(v16 + 16);
        v19 = &src[*(_QWORD *)(v16 + 24)];
        *(_QWORD *)(v16 + 24) = v19;
        result = v21 - (_QWORD)v19;
      }
      if ( result <= 1 )
      {
LABEL_25:
        result = sub_16E7EE0(v16, "]\n", 2u);
        goto LABEL_21;
      }
LABEL_20:
      *(_WORD *)v19 = 2653;
      *(_QWORD *)(v16 + 24) += 2LL;
LABEL_21:
      v6 = (__int64 *)*v6;
      if ( !v6 )
        return result;
    }
    v16 = sub_16E7EE0(v16, v17, v20);
LABEL_24:
    v19 = *(char **)(v16 + 24);
    v22 = *(char **)(v16 + 16);
    v23 = v22 == v19;
    result = v22 - v19;
    if ( v23 || result == 1 )
      goto LABEL_25;
    goto LABEL_20;
  }
  return result;
}
