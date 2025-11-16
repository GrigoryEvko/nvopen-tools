// Function: sub_16E5F30
// Address: 0x16e5f30
//
__int64 __fastcall sub_16E5F30(__int64 a1, char *a2, __int64 a3, __int64 a4, _BYTE *a5, _QWORD *a6)
{
  unsigned int v6; // r14d
  __int64 v7; // r12
  int v9; // eax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __m128i *v17; // rax
  size_t v18; // r14
  unsigned int v19; // r11d
  _QWORD *v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // r11d
  _QWORD *v27; // rcx
  _QWORD *v28; // r9
  __int64 *v29; // rax
  __int64 *v30; // rax
  __m128i *v31; // rdi
  __int64 v32; // rax
  bool v33; // zf
  _QWORD *v34; // [rsp+0h] [rbp-A0h]
  _QWORD *v35; // [rsp+10h] [rbp-90h]
  _QWORD *v36; // [rsp+10h] [rbp-90h]
  unsigned int v38; // [rsp+28h] [rbp-78h]
  unsigned int v39; // [rsp+28h] [rbp-78h]
  char v40; // [rsp+2Ch] [rbp-74h]
  __m128i *v41; // [rsp+50h] [rbp-50h] BYREF
  char *v42; // [rsp+58h] [rbp-48h]
  __m128i v43[4]; // [rsp+60h] [rbp-40h] BYREF

  v6 = 0;
  *a5 = 0;
  v40 = a3;
  if ( *(_DWORD *)(a1 + 96) )
    return v6;
  v7 = *(_QWORD *)(a1 + 264);
  if ( v7 )
  {
    v9 = *(_DWORD *)(*(_QWORD *)(v7 + 8) + 32LL);
    if ( v9 == 4 )
    {
      v12 = -1;
      v41 = v43;
      if ( a2 )
        v12 = (__int64)&a2[strlen(a2)];
      sub_16E3680((__int64 *)&v41, a2, v12);
      v16 = *(unsigned int *)(v7 + 56);
      if ( (unsigned int)v16 >= *(_DWORD *)(v7 + 60) )
      {
        sub_12BE710(v7 + 48, 0, v16, v13, v14, v15);
        LODWORD(v16) = *(_DWORD *)(v7 + 56);
      }
      v17 = (__m128i *)(*(_QWORD *)(v7 + 48) + 32LL * (unsigned int)v16);
      if ( v17 )
      {
        v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
        if ( v41 == v43 )
        {
          v17[1] = _mm_load_si128(v43);
        }
        else
        {
          v17->m128i_i64[0] = (__int64)v41;
          v17[1].m128i_i64[0] = v43[0].m128i_i64[0];
        }
        v17->m128i_i64[1] = (__int64)v42;
        ++*(_DWORD *)(v7 + 56);
      }
      else
      {
        v31 = v41;
        *(_DWORD *)(v7 + 56) = v16 + 1;
        if ( v31 != v43 )
          j_j___libc_free_0(v31, v43[0].m128i_i64[0] + 1);
      }
      v18 = 0;
      if ( a2 )
        v18 = strlen(a2);
      v19 = sub_16D19C0(v7 + 16, (unsigned __int8 *)a2, v18);
      v20 = (_QWORD *)(*(_QWORD *)(v7 + 16) + 8LL * v19);
      v21 = *v20;
      if ( *v20 )
      {
        if ( v21 != -8 )
        {
LABEL_16:
          v22 = *(_QWORD *)(v21 + 8);
          if ( v22 )
          {
            v6 = 1;
            *a6 = *(_QWORD *)(a1 + 264);
            *(_QWORD *)(a1 + 264) = v22;
          }
          else if ( v40 )
          {
            v33 = *a2 == 0;
            v41 = (__m128i *)"missing required key '";
            if ( v33 )
            {
              v43[0].m128i_i16[0] = 259;
            }
            else
            {
              v42 = a2;
              v43[0].m128i_i16[0] = 771;
            }
            v6 = 0;
            sub_16E42A0(a1, *(_QWORD *)(a1 + 264));
          }
          else
          {
            *a5 = 1;
            return 0;
          }
          return v6;
        }
        --*(_DWORD *)(v7 + 32);
      }
      v35 = v20;
      v38 = v19;
      v25 = malloc(v18 + 17);
      v26 = v38;
      v27 = v35;
      v28 = (_QWORD *)v25;
      if ( !v25 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v28 = 0;
        v27 = v35;
        v26 = v38;
      }
      if ( v18 )
      {
        v39 = v26;
        v34 = v28;
        v36 = v27;
        memcpy(v28 + 2, a2, v18);
        v28 = v34;
        v27 = v36;
        v26 = v39;
      }
      *((_BYTE *)v28 + v18 + 16) = 0;
      *v28 = v18;
      v28[1] = 0;
      *v27 = v28;
      ++*(_DWORD *)(v7 + 28);
      v29 = (__int64 *)(*(_QWORD *)(v7 + 16) + 8LL * (unsigned int)sub_16D1CD0(v7 + 16, v26));
      v21 = *v29;
      if ( *v29 == -8 || !v21 )
      {
        v30 = v29 + 1;
        do
        {
          do
            v21 = *v30++;
          while ( v21 == -8 );
        }
        while ( !v21 );
      }
      goto LABEL_16;
    }
    LOBYTE(v6) = a3 | (v9 != 0);
    if ( (_BYTE)v6 )
    {
      v24 = *(_QWORD *)(a1 + 264);
      v6 = 0;
      v43[0].m128i_i16[0] = 259;
      v41 = (__m128i *)"not a mapping";
      sub_16E42A0(a1, v24);
    }
  }
  else if ( (_BYTE)a3 )
  {
    v32 = sub_2241E50(a1, a2, a3, a4, a5);
    *(_DWORD *)(a1 + 96) = 22;
    *(_QWORD *)(a1 + 104) = v32;
  }
  return v6;
}
