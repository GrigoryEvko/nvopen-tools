// Function: sub_1DBBDD0
// Address: 0x1dbbdd0
//
__int64 __fastcall sub_1DBBDD0(__int64 a1, __int64 a2)
{
  __m128i *v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r15
  _BYTE *v7; // rax
  __int64 v8; // rdx
  _BYTE *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rax
  int v12; // r14d
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rdx
  _BYTE *v18; // rax
  __int64 *v19; // rbx
  __int64 *i; // r14
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v25; // [rsp+0h] [rbp-60h]
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  _QWORD v28[2]; // [rsp+10h] [rbp-50h] BYREF
  void (__fastcall *v29)(_QWORD *, _QWORD *, __int64); // [rsp+20h] [rbp-40h]
  void (__fastcall *v30)(_QWORD *, __int64); // [rsp+28h] [rbp-38h]

  v4 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 0x1Fu )
  {
    sub_16E7EE0(a2, "********** INTERVALS **********\n", 0x20u);
  }
  else
  {
    *v4 = _mm_load_si128((const __m128i *)&xmmword_42E9A40);
    v4[1] = _mm_load_si128((const __m128i *)&xmmword_42E9A50);
    *(_QWORD *)(a2 + 24) += 32LL;
  }
  v5 = 0;
  v25 = *(unsigned int *)(a1 + 680);
  if ( (_DWORD)v25 )
  {
    do
    {
      v26 = *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v5);
      if ( v26 )
      {
        sub_1F4AA70(v28, (unsigned int)v5, *(_QWORD *)(a1 + 248));
        if ( !v29 )
          sub_4263D6(v28, (unsigned int)v5, v8);
        v30(v28, a2);
        v9 = *(_BYTE **)(a2 + 24);
        v10 = v26;
        if ( (unsigned __int64)v9 < *(_QWORD *)(a2 + 16) )
        {
          v6 = a2;
          *(_QWORD *)(a2 + 24) = v9 + 1;
          *v9 = 32;
        }
        else
        {
          v11 = sub_16E7DE0(a2, 32);
          v10 = v26;
          v6 = v11;
        }
        sub_1DB50D0(v10, v6);
        v7 = *(_BYTE **)(v6 + 24);
        if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 16) )
        {
          sub_16E7DE0(v6, 10);
        }
        else
        {
          *(_QWORD *)(v6 + 24) = v7 + 1;
          *v7 = 10;
        }
        if ( v29 )
          v29(v28, v28, 3);
      }
      ++v5;
    }
    while ( v25 != v5 );
  }
  v12 = *(_DWORD *)(*(_QWORD *)(a1 + 240) + 32LL);
  if ( v12 )
  {
    v13 = 0;
    while ( 1 )
    {
      v14 = v13 & 0x7FFFFFFF;
      if ( (unsigned int)v14 >= *(_DWORD *)(a1 + 408) )
        goto LABEL_19;
      v15 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v14);
      if ( !v15 )
        goto LABEL_19;
      sub_1DB53F0(v15, a2);
      v16 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v16 < *(_QWORD *)(a2 + 16) )
      {
        *(_QWORD *)(a2 + 24) = v16 + 1;
        *v16 = 10;
LABEL_19:
        if ( ++v13 == v12 )
          break;
      }
      else
      {
        ++v13;
        sub_16E7DE0(a2, 10);
        if ( v13 == v12 )
          break;
      }
    }
  }
  v17 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v17) <= 8 )
  {
    sub_16E7EE0(a2, "RegMasks:", 9u);
    v18 = *(_BYTE **)(a2 + 24);
  }
  else
  {
    *(_BYTE *)(v17 + 8) = 58;
    *(_QWORD *)v17 = 0x736B73614D676552LL;
    v18 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 9LL);
    *(_QWORD *)(a2 + 24) = v18;
  }
  v19 = *(__int64 **)(a1 + 432);
  for ( i = &v19[*(unsigned int *)(a1 + 440)]; i != v19; v18 = *(_BYTE **)(a2 + 24) )
  {
    v22 = *v19;
    if ( (unsigned __int64)v18 < *(_QWORD *)(a2 + 16) )
    {
      v21 = a2;
      *(_QWORD *)(a2 + 24) = v18 + 1;
      *v18 = 32;
    }
    else
    {
      v27 = *v19;
      v23 = sub_16E7DE0(a2, 32);
      v22 = v27;
      v21 = v23;
    }
    ++v19;
    v28[0] = v22;
    sub_1F10810(v28, v21);
  }
  if ( (unsigned __int64)v18 >= *(_QWORD *)(a2 + 16) )
  {
    sub_16E7DE0(a2, 10);
  }
  else
  {
    *(_QWORD *)(a2 + 24) = v18 + 1;
    *v18 = 10;
  }
  return sub_1DBA210(a1, a2);
}
