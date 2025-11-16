// Function: sub_1C2DB50
// Address: 0x1c2db50
//
__int64 __fastcall sub_1C2DB50(__int64 a1, __int64 a2)
{
  char *v3; // rcx
  char *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  char *v7; // rdx
  __int64 v8; // r12
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 *v19; // rax
  __int64 v20; // r13
  _QWORD *v21; // rax
  _BYTE *v22; // rax
  _BYTE *v23; // rcx
  char i; // dl
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  _QWORD *v27; // rbx
  _QWORD *v28; // r12
  __int64 v29; // r14
  __int64 v30; // rdi
  __int64 v31; // rdi
  __m128i *v32; // rdx
  __int64 v33; // rax
  _QWORD *v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // rdi
  _BYTE *v37; // rax

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
    goto LABEL_45;
  if ( v6 == 3 )
  {
    if ( *(_QWORD *)v4 )
      goto LABEL_8;
    v4 += 8;
LABEL_45:
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
    v8 = *(_QWORD *)(*(_QWORD *)v4 + 56LL);
    sub_1649960(v8);
    if ( (unsigned __int8)sub_160E740() )
    {
      v10 = *(__int64 **)(a1 + 8);
      v11 = *v10;
      v12 = v10[1];
      if ( v11 == v12 )
        goto LABEL_60;
      while ( *(_UNKNOWN **)v11 != &unk_4F9E06C )
      {
        v11 += 16;
        if ( v12 == v11 )
          goto LABEL_60;
      }
      v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
              *(_QWORD *)(v11 + 8),
              &unk_4F9E06C);
      v14 = *(__int64 **)(a1 + 8);
      v15 = v13 + 160;
      v16 = *v14;
      v17 = v14[1];
      if ( v16 == v17 )
LABEL_60:
        BUG();
      while ( *(_UNKNOWN **)v16 != &unk_4F9920C )
      {
        v16 += 16;
        if ( v17 == v16 )
          goto LABEL_60;
      }
      v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(
              *(_QWORD *)(v16 + 8),
              &unk_4F9920C)
          + 160;
      v19 = (__int64 *)sub_22077B0(280);
      v20 = (__int64)v19;
      if ( v19 )
      {
        *v19 = v8;
        v21 = v19 + 27;
        *(v21 - 26) = v18;
        *(v21 - 25) = v15;
        *(v21 - 24) = 0;
        *(v21 - 23) = 0;
        *(v21 - 22) = 0;
        *((_BYTE *)v21 - 168) = 0;
        *(v21 - 20) = 0;
        *(v21 - 19) = 0;
        *(v21 - 18) = 0;
        *((_DWORD *)v21 - 34) = 0;
        *(v21 - 16) = 0;
        *(v21 - 15) = 0;
        *(v21 - 14) = 0;
        *(v21 - 13) = 0;
        *(v21 - 12) = 0;
        *(v21 - 11) = 0;
        *((_DWORD *)v21 - 20) = 0;
        *(v21 - 9) = 0;
        *(v21 - 8) = 0;
        *(v21 - 7) = 0;
        *((_DWORD *)v21 - 12) = 0;
        *(_QWORD *)(v20 + 176) = 0;
        *(_QWORD *)(v20 + 184) = v21;
        *(_QWORD *)(v20 + 192) = v21;
        *(_QWORD *)(v20 + 200) = 8;
        *(_DWORD *)(v20 + 208) = 0;
      }
      sub_1C2D330(v20);
      v22 = (_BYTE *)qword_4F9E580[20];
      v23 = (_BYTE *)qword_4F9E580[21];
      for ( i = 0; v23 != v22; ++v22 )
        i |= *v22;
      if ( (i & 1) != 0 )
      {
        v31 = *(_QWORD *)(a1 + 160);
        v32 = *(__m128i **)(v31 + 24);
        if ( *(_QWORD *)(v31 + 16) - (_QWORD)v32 <= 0xFu )
        {
          v31 = sub_16E7EE0(v31, "Max Live RRegs: ", 0x10u);
        }
        else
        {
          *v32 = _mm_load_si128((const __m128i *)&xmmword_3F6EF60);
          *(_QWORD *)(v31 + 24) += 16LL;
        }
        v33 = sub_16E7AB0(v31, *(int *)(v20 + 24));
        v34 = *(_QWORD **)(v33 + 24);
        v35 = v33;
        if ( *(_QWORD *)(v33 + 16) - (_QWORD)v34 <= 7u )
        {
          v35 = sub_16E7EE0(v33, "\tPRegs: ", 8u);
        }
        else
        {
          *v34 = 0x203A736765525009LL;
          *(_QWORD *)(v33 + 24) += 8LL;
        }
        v36 = sub_16E7AB0(v35, *(int *)(v20 + 28));
        v37 = *(_BYTE **)(v36 + 24);
        if ( *(_BYTE **)(v36 + 16) == v37 )
        {
          sub_16E7EE0(v36, "\t", 1u);
        }
        else
        {
          *v37 = 9;
          ++*(_QWORD *)(v36 + 24);
        }
        sub_1C28B20((__int64 *)(a1 + 160), v8);
        sub_1C28930((__int64 *)(a1 + 160), *(char **)(a1 + 168), *(_QWORD *)(a1 + 176), v8);
      }
      else
      {
        sub_1C28B20((__int64 *)(a1 + 160), v8);
        sub_1C28930((__int64 *)(a1 + 160), *(char **)(a1 + 168), *(_QWORD *)(a1 + 176), v8);
        if ( !v20 )
          return 0;
      }
      v25 = *(_QWORD *)(v20 + 192);
      if ( v25 != *(_QWORD *)(v20 + 184) )
        _libc_free(v25);
      j___libc_free_0(*(_QWORD *)(v20 + 152));
      v26 = *(unsigned int *)(v20 + 136);
      if ( (_DWORD)v26 )
      {
        v27 = *(_QWORD **)(v20 + 120);
        v28 = &v27[2 * v26];
        do
        {
          if ( *v27 != -16 && *v27 != -8 )
          {
            v29 = v27[1];
            if ( v29 )
            {
              _libc_free(*(_QWORD *)(v29 + 48));
              _libc_free(*(_QWORD *)(v29 + 24));
              j_j___libc_free_0(v29, 72);
            }
          }
          v27 += 2;
        }
        while ( v28 != v27 );
      }
      j___libc_free_0(*(_QWORD *)(v20 + 120));
      v30 = *(_QWORD *)(v20 + 88);
      if ( v30 )
        j_j___libc_free_0(v30, *(_QWORD *)(v20 + 104) - v30);
      j___libc_free_0(*(_QWORD *)(v20 + 64));
      j_j___libc_free_0(v20, 280);
    }
  }
  return 0;
}
