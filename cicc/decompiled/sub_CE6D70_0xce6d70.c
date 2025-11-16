// Function: sub_CE6D70
// Address: 0xce6d70
//
__int64 __fastcall sub_CE6D70(_QWORD *a1, __int64 a2)
{
  char *v3; // rcx
  char *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  char *v7; // rdx
  __int64 v8; // r13
  char *v9; // rax
  __int64 v10; // rdx
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 *v21; // rax
  __int64 *v22; // r12
  _QWORD *v23; // rax
  _BYTE *v24; // rax
  char v25; // dl
  unsigned __int8 *v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rax
  _QWORD *v29; // rbx
  _QWORD *v30; // r13
  __int64 v31; // r14
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rdi
  __m128i *v36; // rdx
  __int64 v37; // rax
  _QWORD *v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rdi
  _BYTE *v41; // rax

  v3 = *(char **)(a2 + 40);
  v4 = *(char **)(a2 + 32);
  v5 = (v3 - v4) >> 5;
  v6 = (v3 - v4) >> 3;
  if ( v5 > 0 )
  {
    v7 = &v4[32 * v5];
    while ( !*(_QWORD *)v4 )
    {
      if ( *((_QWORD *)v4 + 1) )
      {
        v4 += 8;
        goto LABEL_8;
      }
      if ( *((_QWORD *)v4 + 2) )
      {
        v4 += 16;
        goto LABEL_8;
      }
      if ( *((_QWORD *)v4 + 3) )
      {
        v4 += 24;
        goto LABEL_8;
      }
      v4 += 32;
      if ( v7 == v4 )
      {
        v6 = (v3 - v4) >> 3;
        goto LABEL_12;
      }
    }
    goto LABEL_8;
  }
LABEL_12:
  if ( v6 == 2 )
    goto LABEL_50;
  if ( v6 == 3 )
  {
    if ( *(_QWORD *)v4 )
      goto LABEL_8;
    v4 += 8;
LABEL_50:
    if ( *(_QWORD *)v4 )
      goto LABEL_8;
    v4 += 8;
    goto LABEL_15;
  }
  if ( v6 != 1 )
    return 0;
LABEL_15:
  if ( !*(_QWORD *)v4 )
    return 0;
LABEL_8:
  if ( v3 != v4 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)v4 + 72LL);
    v9 = (char *)sub_BD5D20(v8);
    if ( sub_BC63A0(v9, v10) )
    {
      v12 = (__int64 *)a1[1];
      v13 = *v12;
      v14 = v12[1];
      if ( v13 == v14 )
        goto LABEL_65;
      while ( *(_UNKNOWN **)v13 != &unk_4F8144C )
      {
        v13 += 16;
        if ( v14 == v13 )
          goto LABEL_65;
      }
      v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
              *(_QWORD *)(v13 + 8),
              &unk_4F8144C);
      v16 = (__int64 *)a1[1];
      v17 = v15 + 176;
      v18 = *v16;
      v19 = v16[1];
      if ( v18 == v19 )
LABEL_65:
        BUG();
      while ( *(_UNKNOWN **)v18 != &unk_4F875EC )
      {
        v18 += 16;
        if ( v19 == v18 )
          goto LABEL_65;
      }
      v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(
              *(_QWORD *)(v18 + 8),
              &unk_4F875EC)
          + 176;
      v21 = (__int64 *)sub_22077B0(272);
      v22 = v21;
      if ( v21 )
      {
        *v21 = v8;
        v23 = v21 + 26;
        *(v23 - 25) = v20;
        *(v23 - 24) = v17;
        *(v23 - 23) = 0;
        *(v23 - 22) = 0;
        *(v23 - 21) = 0;
        *((_BYTE *)v23 - 160) = 0;
        *(v23 - 19) = 0;
        *(v23 - 18) = 0;
        *(v23 - 17) = 0;
        *((_DWORD *)v23 - 32) = 0;
        *(v23 - 15) = 0;
        *(v23 - 14) = 0;
        *(v23 - 13) = 0;
        *(v23 - 12) = 0;
        *(v23 - 11) = 0;
        *(v23 - 10) = 0;
        *((_DWORD *)v23 - 18) = 0;
        *(v23 - 8) = 0;
        *(v23 - 7) = 0;
        *(v23 - 6) = 0;
        *((_DWORD *)v23 - 10) = 0;
        *(v23 - 4) = 0;
        v22[23] = (__int64)v23;
        v22[24] = 8;
        *((_DWORD *)v22 + 50) = 0;
        *((_BYTE *)v22 + 204) = 1;
      }
      sub_CE6510(v22);
      v24 = (_BYTE *)unk_4F83008;
      v25 = 0;
      if ( unk_4F83008 != unk_4F83010 )
      {
        do
          v25 |= *v24++;
        while ( (_BYTE *)unk_4F83010 != v24 );
      }
      if ( (v25 & 1) != 0 )
      {
        v35 = a1[22];
        v36 = *(__m128i **)(v35 + 32);
        if ( *(_QWORD *)(v35 + 24) - (_QWORD)v36 <= 0xFu )
        {
          v35 = sub_CB6200(v35, "Max Live RRegs: ", 0x10u);
        }
        else
        {
          *v36 = _mm_load_si128((const __m128i *)&xmmword_3F6EF60);
          *(_QWORD *)(v35 + 32) += 16LL;
        }
        v37 = sub_CB59F0(v35, *((int *)v22 + 6));
        v38 = *(_QWORD **)(v37 + 32);
        v39 = v37;
        if ( *(_QWORD *)(v37 + 24) - (_QWORD)v38 <= 7u )
        {
          v39 = sub_CB6200(v37, "\tPRegs: ", 8u);
        }
        else
        {
          *v38 = 0x203A736765525009LL;
          *(_QWORD *)(v37 + 32) += 8LL;
        }
        v40 = sub_CB59F0(v39, *((int *)v22 + 7));
        v41 = *(_BYTE **)(v40 + 32);
        if ( *(_BYTE **)(v40 + 24) == v41 )
        {
          sub_CB6200(v40, (unsigned __int8 *)"\t", 1u);
        }
        else
        {
          *v41 = 9;
          ++*(_QWORD *)(v40 + 32);
        }
        sub_CE1CE0(a1 + 22, v8);
        v26 = (unsigned __int8 *)a1[23];
        sub_CE19D0(a1 + 22, v26, a1[24], v8);
      }
      else
      {
        sub_CE1CE0(a1 + 22, v8);
        v26 = (unsigned __int8 *)a1[23];
        sub_CE19D0(a1 + 22, v26, a1[24], v8);
        if ( !v22 )
          return 0;
      }
      if ( !*((_BYTE *)v22 + 204) )
        _libc_free(v22[23], v26);
      v27 = 16LL * *((unsigned int *)v22 + 42);
      sub_C7D6A0(v22[19], v27, 8);
      v28 = *((unsigned int *)v22 + 34);
      if ( (_DWORD)v28 )
      {
        v29 = (_QWORD *)v22[15];
        v30 = &v29[2 * v28];
        do
        {
          if ( *v29 != -8192 && *v29 != -4096 )
          {
            v31 = v29[1];
            if ( v31 )
            {
              v32 = *(_QWORD *)(v31 + 96);
              if ( v32 != v31 + 112 )
                _libc_free(v32, v27);
              v33 = *(_QWORD *)(v31 + 24);
              if ( v33 != v31 + 40 )
                _libc_free(v33, v27);
              v27 = 168;
              j_j___libc_free_0(v31, 168);
            }
          }
          v29 += 2;
        }
        while ( v30 != v29 );
        LODWORD(v28) = *((_DWORD *)v22 + 34);
      }
      sub_C7D6A0(v22[15], 16LL * (unsigned int)v28, 8);
      v34 = v22[11];
      if ( v34 )
        j_j___libc_free_0(v34, v22[13] - v34);
      sub_C7D6A0(v22[8], 16LL * *((unsigned int *)v22 + 20), 8);
      j_j___libc_free_0(v22, 272);
    }
  }
  return 0;
}
