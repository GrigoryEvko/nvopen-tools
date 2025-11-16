// Function: sub_E00CC0
// Address: 0xe00cc0
//
__m128i *__fastcall sub_E00CC0(__m128i *a1, const __m128i *a2, unsigned int a3)
{
  __m128i v3; // xmm1
  bool v4; // zf
  __int64 v5; // rbx
  unsigned __int8 v7; // r14
  __int64 *v8; // r15
  _BYTE *v9; // r8
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rbx
  unsigned int v15; // r13d
  __int64 v16; // r14
  __int64 v17; // rax
  int v18; // eax
  unsigned int v19; // [rsp+Ch] [rbp-44h]
  _BYTE *v20; // [rsp+10h] [rbp-40h]
  int v21; // [rsp+18h] [rbp-38h]
  _BYTE *v22; // [rsp+18h] [rbp-38h]
  _BYTE *v23; // [rsp+18h] [rbp-38h]

  v3 = _mm_loadu_si128(a2 + 1);
  *a1 = _mm_loadu_si128(a2);
  v4 = a1->m128i_i64[0] == 0;
  v5 = a1->m128i_i64[1];
  a1[1] = v3;
  if ( v4 && v5 )
  {
    v7 = *(_BYTE *)(v5 - 16);
    if ( (v7 & 2) != 0 )
    {
      if ( *(_DWORD *)(v5 - 24) <= 2u )
        goto LABEL_3;
      v8 = *(__int64 **)(v5 - 32);
      v9 = (_BYTE *)(v5 - 16);
      v10 = *v8;
      if ( !*v8 )
        goto LABEL_3;
    }
    else
    {
      if ( ((*(_WORD *)(v5 - 16) >> 6) & 0xFu) <= 2 )
        goto LABEL_3;
      v9 = (_BYTE *)(v5 - 16);
      v8 = (__int64 *)(v5 - 16 - 8LL * ((v7 >> 2) & 0xF));
      v10 = *v8;
      if ( !*v8 )
        goto LABEL_3;
    }
    if ( *(_BYTE *)v10 == 1 )
    {
      v11 = *(_QWORD *)(v10 + 136);
      if ( *(_BYTE *)v11 == 17 )
      {
        if ( *(_DWORD *)(v11 + 32) <= 0x40u )
        {
          if ( *(_QWORD *)(v11 + 24) )
            goto LABEL_3;
        }
        else
        {
          v19 = a3;
          v20 = v9;
          v21 = *(_DWORD *)(v11 + 32);
          v12 = sub_C444A0(v11 + 24);
          v9 = v20;
          a3 = v19;
          if ( v21 != v12 )
            goto LABEL_3;
        }
        if ( (v7 & 2) != 0 )
        {
          v13 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 8LL);
          if ( !v13 )
            goto LABEL_3;
        }
        else
        {
          v13 = *(_QWORD *)&v9[-8 * ((v7 >> 2) & 0xF) + 8];
          if ( !v13 )
            goto LABEL_3;
        }
        if ( *(_BYTE *)v13 == 1 )
        {
          v14 = *(_QWORD *)(v13 + 136);
          if ( *(_BYTE *)v14 == 17 )
          {
            v15 = *(_DWORD *)(v14 + 32);
            v16 = a3;
            if ( v15 > 0x40 )
            {
              v23 = v9;
              v18 = sub_C444A0(v14 + 24);
              v9 = v23;
              if ( v15 - v18 > 0x40 )
                goto LABEL_3;
              v17 = **(_QWORD **)(v14 + 24);
            }
            else
            {
              v17 = *(_QWORD *)(v14 + 24);
            }
            if ( v16 == v17 )
            {
              if ( v8[2] )
              {
                v22 = v9;
                if ( (unsigned __int8)(**((_BYTE **)sub_A17150(v9) + 2) - 5) <= 0x1Fu )
                  a1->m128i_i64[0] = *((_QWORD *)sub_A17150(v22) + 2);
              }
            }
          }
        }
      }
    }
  }
LABEL_3:
  a1->m128i_i64[1] = 0;
  return a1;
}
