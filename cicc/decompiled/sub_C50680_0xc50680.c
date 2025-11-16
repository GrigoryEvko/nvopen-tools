// Function: sub_C50680
// Address: 0xc50680
//
__int64 __fastcall sub_C50680(__int64 a1, __int64 a2)
{
  __int64 *v4; // rsi
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 **v9; // r14
  _QWORD *v10; // rax
  __int64 *v11; // r12
  __int64 *v12; // r13
  unsigned int v13; // eax
  int v14; // eax
  __int64 result; // rax
  int v16; // ecx
  __int64 *v17; // rdx
  __int64 *v18; // rdi
  __int64 *v19; // rdx
  int v20; // ecx
  __int64 *v21; // rdi
  __int64 v22; // rax
  __m128i v23; // xmm0
  unsigned __int64 v24; // rdx
  _QWORD *v25; // [rsp+0h] [rbp-170h]
  _QWORD *v26; // [rsp+10h] [rbp-160h]
  __int64 **i; // [rsp+18h] [rbp-158h]
  __m128i v28; // [rsp+20h] [rbp-150h] BYREF
  __int64 **v29; // [rsp+30h] [rbp-140h] BYREF
  __int64 v30; // [rsp+38h] [rbp-138h]
  _BYTE v31[304]; // [rsp+40h] [rbp-130h] BYREF

  v4 = (__int64 *)&v29;
  v29 = (__int64 **)v31;
  v30 = 0x1000000000LL;
  (*(void (__fastcall **)(__int64, __int64 ***))(*(_QWORD *)a1 + 72LL))(a1, &v29);
  if ( *(_QWORD *)(a1 + 32) )
  {
    v22 = (unsigned int)v30;
    v23 = _mm_loadu_si128((const __m128i *)(a1 + 24));
    v24 = (unsigned int)v30 + 1LL;
    if ( v24 > HIDWORD(v30) )
    {
      v4 = (__int64 *)v31;
      v28 = v23;
      sub_C8D5F0(&v29, v31, v24, 16);
      v22 = (unsigned int)v30;
      v23 = _mm_load_si128(&v28);
    }
    *(__m128i *)&v29[2 * v22] = v23;
    v5 = (unsigned int)(v30 + 1);
    LODWORD(v30) = v30 + 1;
  }
  else
  {
    v5 = (unsigned int)v30;
  }
  v6 = *(unsigned int *)(a2 + 136);
  v7 = *(_QWORD *)(a2 + 128);
  v8 = 2 * v5;
  v28.m128i_i64[0] = a2 + 128;
  v26 = (_QWORD *)(v7 + 8 * v6);
  v9 = v29;
  for ( i = &v29[v8]; i != v9; v9 += 2 )
  {
    v11 = v9[1];
    v12 = *v9;
    v13 = sub_C92610(*v9, v11);
    v4 = v12;
    v14 = sub_C92860(v28.m128i_i64[0], v12, v11, v13);
    if ( v14 == -1 )
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 128) + 8LL * *(unsigned int *)(a2 + 136));
    else
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 128) + 8LL * v14);
    if ( v10 != v26 && a1 == *(_QWORD *)(*v10 + 8LL) )
    {
      v25 = (_QWORD *)*v10;
      sub_C929B0(v28.m128i_i64[0], *v10);
      v4 = (__int64 *)(*v25 + 17LL);
      sub_C7D6A0(v25, v4, 8);
    }
  }
  result = (*(_WORD *)(a1 + 12) >> 7) & 3;
  if ( ((*(_WORD *)(a1 + 12) >> 7) & 3) == 1 )
  {
    result = *(unsigned int *)(a2 + 40);
    v4 = *(__int64 **)(a2 + 32);
    v19 = &v4[result];
    v20 = *(_DWORD *)(a2 + 40);
    if ( v4 != v19 )
    {
      while ( 1 )
      {
        result = *v4;
        v21 = v4++;
        if ( a1 == result )
          break;
        if ( v19 == v4 )
          goto LABEL_16;
      }
      if ( v19 != v4 )
      {
        result = (__int64)memmove(v21, v4, (char *)v19 - (char *)v4);
        v20 = *(_DWORD *)(a2 + 40);
      }
      *(_DWORD *)(a2 + 40) = v20 - 1;
    }
  }
  else if ( (*(_BYTE *)(a1 + 13) & 8) != 0 )
  {
    result = *(unsigned int *)(a2 + 88);
    v4 = *(__int64 **)(a2 + 80);
    v16 = *(_DWORD *)(a2 + 88);
    v17 = &v4[result];
    while ( v4 != v17 )
    {
      result = *v4;
      v18 = v4++;
      if ( a1 == result )
      {
        if ( v4 != v17 )
        {
          result = (__int64)memmove(v18, v4, (char *)v17 - (char *)v4);
          v16 = *(_DWORD *)(a2 + 88);
        }
        *(_DWORD *)(a2 + 88) = v16 - 1;
        break;
      }
    }
  }
  else if ( a1 == *(_QWORD *)(a2 + 152) )
  {
    *(_QWORD *)(a2 + 152) = 0;
  }
LABEL_16:
  if ( v29 != (__int64 **)v31 )
    return _libc_free(v29, v4);
  return result;
}
