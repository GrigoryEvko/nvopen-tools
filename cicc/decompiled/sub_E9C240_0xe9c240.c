// Function: sub_E9C240
// Address: 0xe9c240
//
__int64 *__fastcall sub_E9C240(__int64 *a1, char *a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  bool v6; // cf
  unsigned __int64 v7; // rax
  char *v8; // rdx
  __int64 v9; // rbx
  char *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  char *v16; // r15
  __int64 i; // rbx
  int v18; // edx
  _QWORD *v19; // r12
  _QWORD *v20; // r14
  _QWORD *v21; // rdi
  __int64 v22; // rdi
  char *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  char *v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  char *v33; // [rsp+28h] [rbp-38h]

  v33 = (char *)a1[1];
  v31 = (char *)*a1;
  v4 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v33[-*a1] >> 5);
  if ( v4 == 0x155555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v33[-*a1] >> 5);
  v6 = __CFADD__(v5, v4);
  v7 = v5 - 0x5555555555555555LL * ((__int64)&v33[-*a1] >> 5);
  v8 = (char *)(a2 - v31);
  if ( v6 )
  {
    v27 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v7 )
    {
      v29 = 0;
      v9 = 96;
      v32 = 0;
      goto LABEL_7;
    }
    if ( v7 > 0x155555555555555LL )
      v7 = 0x155555555555555LL;
    v27 = 96 * v7;
  }
  v28 = sub_22077B0(v27);
  v8 = (char *)(a2 - v31);
  v32 = v28;
  v29 = v28 + v27;
  v9 = v28 + 96;
LABEL_7:
  v10 = &v8[v32];
  if ( &v8[v32] )
  {
    v11 = *(_QWORD *)(a3 + 56);
    *(_QWORD *)v10 = *(_QWORD *)a3;
    v12 = *(_QWORD *)(a3 + 8);
    *((_QWORD *)v10 + 7) = v11;
    *((_QWORD *)v10 + 1) = v12;
    *((_QWORD *)v10 + 2) = *(_QWORD *)(a3 + 16);
    *((_QWORD *)v10 + 3) = *(_QWORD *)(a3 + 24);
    v13 = *(_QWORD *)(a3 + 32);
    *(_QWORD *)(a3 + 32) = 0;
    *((_QWORD *)v10 + 4) = v13;
    v14 = *(_QWORD *)(a3 + 40);
    *(_QWORD *)(a3 + 40) = 0;
    *((_QWORD *)v10 + 5) = v14;
    v15 = *(_QWORD *)(a3 + 48);
    *(_QWORD *)(a3 + 48) = 0;
    *((_QWORD *)v10 + 6) = v15;
    *((_DWORD *)v10 + 16) = *(_DWORD *)(a3 + 64);
    *((_QWORD *)v10 + 9) = *(_QWORD *)(a3 + 72);
    *((_WORD *)v10 + 40) = *(_WORD *)(a3 + 80);
    *((_DWORD *)v10 + 21) = *(_DWORD *)(a3 + 84);
    *((_WORD *)v10 + 44) = *(_WORD *)(a3 + 88);
  }
  v16 = v31;
  if ( a2 != v31 )
  {
    for ( i = v32; ; i += 96 )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v16;
        *(_QWORD *)(i + 8) = *((_QWORD *)v16 + 1);
        *(_QWORD *)(i + 16) = *((_QWORD *)v16 + 2);
        *(_QWORD *)(i + 24) = *((_QWORD *)v16 + 3);
        *(_QWORD *)(i + 32) = *((_QWORD *)v16 + 4);
        *(_QWORD *)(i + 40) = *((_QWORD *)v16 + 5);
        *(_QWORD *)(i + 48) = *((_QWORD *)v16 + 6);
        v18 = *((_DWORD *)v16 + 14);
        *((_QWORD *)v16 + 6) = 0;
        *((_QWORD *)v16 + 5) = 0;
        *((_QWORD *)v16 + 4) = 0;
        *(_DWORD *)(i + 56) = v18;
        *(_DWORD *)(i + 60) = *((_DWORD *)v16 + 15);
        *(_DWORD *)(i + 64) = *((_DWORD *)v16 + 16);
        *(_QWORD *)(i + 72) = *((_QWORD *)v16 + 9);
        *(_BYTE *)(i + 80) = v16[80];
        *(_BYTE *)(i + 81) = v16[81];
        *(_DWORD *)(i + 84) = *((_DWORD *)v16 + 21);
        *(_BYTE *)(i + 88) = v16[88];
        *(_BYTE *)(i + 89) = v16[89];
      }
      v19 = (_QWORD *)*((_QWORD *)v16 + 5);
      v20 = (_QWORD *)*((_QWORD *)v16 + 4);
      if ( v19 != v20 )
      {
        do
        {
          v21 = (_QWORD *)v20[9];
          if ( v21 != v20 + 11 )
            j_j___libc_free_0(v21, v20[11] + 1LL);
          v22 = v20[6];
          if ( v22 )
            j_j___libc_free_0(v22, v20[8] - v22);
          v20 += 13;
        }
        while ( v19 != v20 );
        v20 = (_QWORD *)*((_QWORD *)v16 + 4);
      }
      if ( v20 )
        j_j___libc_free_0(v20, *((_QWORD *)v16 + 6) - (_QWORD)v20);
      v16 += 96;
      if ( v16 == a2 )
        break;
    }
    v9 = i + 192;
  }
  if ( a2 != v33 )
  {
    v23 = a2;
    v24 = v9;
    do
    {
      v25 = *(_QWORD *)v23;
      v24 += 96;
      v23 += 96;
      *(_QWORD *)(v24 - 96) = v25;
      *(_QWORD *)(v24 - 88) = *((_QWORD *)v23 - 11);
      *(_QWORD *)(v24 - 80) = *((_QWORD *)v23 - 10);
      *(_QWORD *)(v24 - 72) = *((_QWORD *)v23 - 9);
      *(_QWORD *)(v24 - 64) = *((_QWORD *)v23 - 8);
      *(_QWORD *)(v24 - 56) = *((_QWORD *)v23 - 7);
      *(_QWORD *)(v24 - 48) = *((_QWORD *)v23 - 6);
      *(_DWORD *)(v24 - 40) = *((_DWORD *)v23 - 10);
      *(_DWORD *)(v24 - 36) = *((_DWORD *)v23 - 9);
      *(_DWORD *)(v24 - 32) = *((_DWORD *)v23 - 8);
      *(_QWORD *)(v24 - 24) = *((_QWORD *)v23 - 3);
      *(_BYTE *)(v24 - 16) = *(v23 - 16);
      *(_BYTE *)(v24 - 15) = *(v23 - 15);
      *(_DWORD *)(v24 - 12) = *((_DWORD *)v23 - 3);
      *(_BYTE *)(v24 - 8) = *(v23 - 8);
      *(_BYTE *)(v24 - 7) = *(v23 - 7);
    }
    while ( v23 != v33 );
    v9 += 32 * (3 * ((0x2AAAAAAAAAAAAABLL * ((unsigned __int64)(v23 - a2 - 96) >> 5)) & 0x7FFFFFFFFFFFFFFLL) + 3);
  }
  if ( v31 )
    j_j___libc_free_0(v31, a1[2] - (_QWORD)v31);
  a1[1] = v9;
  *a1 = v32;
  a1[2] = v29;
  return a1;
}
