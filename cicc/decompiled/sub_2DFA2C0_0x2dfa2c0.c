// Function: sub_2DFA2C0
// Address: 0x2dfa2c0
//
void __fastcall sub_2DFA2C0(__int64 a1)
{
  __int64 v2; // rcx
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // r9
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rdx
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // r14
  unsigned __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // r12
  __int64 v21; // r13
  unsigned __int64 v22; // rdi
  __int64 *v23; // r14
  __int64 v24; // rax
  __int64 *v25; // r12
  __int64 *i; // rax
  __int64 v27; // rdi
  unsigned int v28; // ecx
  __int64 v29; // rsi
  __int64 *v30; // r12
  unsigned __int64 v31; // r13
  __int64 v32; // rsi
  __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  _QWORD *v35; // rdi
  __int64 v36; // [rsp+8h] [rbp-38h]

  sub_C7D6A0(*(_QWORD *)(a1 + 1152), 48LL * *(unsigned int *)(a1 + 1168), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1120), 16LL * *(unsigned int *)(a1 + 1136), 8);
  v5 = *(_QWORD *)(a1 + 1080);
  v6 = v5 + 8LL * *(unsigned int *)(a1 + 1088);
  if ( v5 != v6 )
  {
    do
    {
      v7 = *(_QWORD *)(v6 - 8);
      v6 -= 8LL;
      if ( v7 )
      {
        v8 = *(_QWORD *)(v7 + 8);
        if ( v8 )
          sub_B91220(v7 + 8, v8);
        j_j___libc_free_0(v7);
      }
    }
    while ( v5 != v6 );
    v6 = *(_QWORD *)(a1 + 1080);
  }
  if ( v6 != a1 + 1096 )
    _libc_free(v6);
  v9 = *(_QWORD *)(a1 + 1000);
  v10 = v9 + 8LL * *(unsigned int *)(a1 + 1008);
  v36 = v9;
  if ( v9 != v10 )
  {
    do
    {
      v11 = *(_QWORD *)(v10 - 8);
      v10 -= 8LL;
      if ( v11 )
      {
        v12 = *(_QWORD *)(v11 + 456);
        while ( v12 )
        {
          sub_2DF5850(*(_QWORD *)(v12 + 24));
          v13 = v12;
          v12 = *(_QWORD *)(v12 + 16);
          j_j___libc_free_0(v13);
        }
        v14 = *(_QWORD *)(v11 + 408);
        if ( v14 != v11 + 424 )
          _libc_free(v14);
        if ( *(_DWORD *)(v11 + 392) )
        {
          sub_2DF5350(v11 + 232, (__int64)sub_2DF57F0, 0, v2, v3, v4);
          *(_DWORD *)(v11 + 392) = 0;
          memset((void *)(v11 + 232), 0, 0xA0u);
          v35 = (_QWORD *)(v11 + 232);
          do
          {
            *v35 = 0;
            v35 += 2;
            *(v35 - 1) = 0;
          }
          while ( v35 != (_QWORD *)(v11 + 296) );
          do
          {
            *v35 = 0;
            v35 += 3;
            *((_BYTE *)v35 - 16) = 0;
            *(v35 - 1) = 0;
          }
          while ( (_QWORD *)(v11 + 392) != v35 );
        }
        v15 = (unsigned __int64 *)(v11 + 368);
        *(_DWORD *)(v11 + 396) = 0;
        do
        {
          if ( *v15 )
            j_j___libc_free_0_0(*v15);
          v15 -= 3;
        }
        while ( (unsigned __int64 *)(v11 + 272) != v15 );
        v16 = *(_QWORD *)(v11 + 56);
        if ( v16 != v11 + 72 )
          _libc_free(v16);
        v17 = *(_QWORD *)(v11 + 32);
        if ( v17 )
          sub_B91220(v11 + 32, v17);
        j_j___libc_free_0(v11);
      }
    }
    while ( v36 != v10 );
    v10 = *(_QWORD *)(a1 + 1000);
  }
  if ( v10 != a1 + 1016 )
    _libc_free(v10);
  v18 = *(_QWORD *)(a1 + 208);
  if ( v18 != a1 + 224 )
    _libc_free(v18);
  v19 = *(unsigned int *)(a1 + 200);
  if ( (_DWORD)v19 )
  {
    v20 = *(_QWORD *)(a1 + 184);
    v21 = v20 + 32 * v19;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v20 <= 0xFFFFFFFD )
        {
          v22 = *(_QWORD *)(v20 + 8);
          if ( v22 )
            break;
        }
        v20 += 32;
        if ( v21 == v20 )
          goto LABEL_38;
      }
      v20 += 32;
      j_j___libc_free_0(v22);
    }
    while ( v21 != v20 );
LABEL_38:
    v19 = *(unsigned int *)(a1 + 200);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 184), 32 * v19, 8);
  sub_2DF5A20(*(_QWORD *)(a1 + 144));
  v23 = *(__int64 **)(a1 + 24);
  v24 = *(unsigned int *)(a1 + 32);
  *(_QWORD *)a1 = 0;
  v25 = &v23[v24];
  if ( v23 != v25 )
  {
    for ( i = v23; ; i = *(__int64 **)(a1 + 24) )
    {
      v27 = *v23;
      v28 = (unsigned int)(v23 - i) >> 7;
      v29 = 4096LL << v28;
      if ( v28 >= 0x1E )
        v29 = 0x40000000000LL;
      ++v23;
      sub_C7D6A0(v27, v29, 16);
      if ( v25 == v23 )
        break;
    }
  }
  v30 = *(__int64 **)(a1 + 72);
  v31 = (unsigned __int64)&v30[2 * *(unsigned int *)(a1 + 80)];
  if ( v30 != (__int64 *)v31 )
  {
    do
    {
      v32 = v30[1];
      v33 = *v30;
      v30 += 2;
      sub_C7D6A0(v33, v32, 16);
    }
    while ( (__int64 *)v31 != v30 );
    v31 = *(_QWORD *)(a1 + 72);
  }
  if ( v31 != a1 + 88 )
    _libc_free(v31);
  v34 = *(_QWORD *)(a1 + 24);
  if ( v34 != a1 + 40 )
    _libc_free(v34);
}
