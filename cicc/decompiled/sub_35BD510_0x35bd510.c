// Function: sub_35BD510
// Address: 0x35bd510
//
unsigned __int64 __fastcall sub_35BD510(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  _QWORD *v4; // r12
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // rsi
  __int64 v11; // rcx
  char *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rsi
  unsigned __int64 v22; // r15
  __int64 i; // r14
  __int64 v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx
  volatile signed __int32 *v28; // rdi
  signed __int32 v29; // ecx
  unsigned __int64 v30; // rdi
  volatile signed __int32 *v31; // rdi
  unsigned __int64 v32; // rdi
  signed __int32 v33; // ecx
  _QWORD *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  unsigned __int64 v37; // rdi
  unsigned __int64 v39; // rcx
  __int64 *v40; // [rsp+0h] [rbp-60h]
  unsigned __int64 v41; // [rsp+10h] [rbp-50h]
  unsigned __int64 v42; // [rsp+18h] [rbp-48h]
  unsigned __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  v4 = (_QWORD *)a1[1];
  v43 = *a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v4 - *a1) >> 5);
  if ( v5 == 0x155555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * ((a1[1] - *a1) >> 5);
  v8 = __CFADD__(v6, v5);
  v9 = v6 - 0x5555555555555555LL * ((a1[1] - *a1) >> 5);
  v10 = (char *)a2 - v43;
  if ( v8 )
  {
    v39 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v41 = 0;
      v11 = 96;
      v45 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x155555555555555LL )
      v9 = 0x155555555555555LL;
    v39 = 96 * v9;
  }
  v40 = a3;
  v42 = v39;
  v45 = sub_22077B0(v39);
  a3 = v40;
  v41 = v45 + v42;
  v11 = v45 + 96;
LABEL_7:
  v12 = &v10[v45];
  if ( &v10[v45] )
  {
    v13 = *a3;
    v14 = a3[2];
    *a3 = 0;
    *(_QWORD *)v12 = v13;
    v15 = a3[1];
    *((_QWORD *)v12 + 2) = v14;
    *((_QWORD *)v12 + 1) = v15;
    LODWORD(v15) = *((_DWORD *)a3 + 6);
    a3[1] = 0;
    *((_DWORD *)v12 + 6) = v15;
    v16 = a3[4];
    a3[4] = 0;
    *((_QWORD *)v12 + 4) = v16;
    *((_DWORD *)v12 + 10) = *((_DWORD *)a3 + 10);
    v17 = a3[6];
    a3[6] = 0;
    *((_QWORD *)v12 + 6) = v17;
    v18 = a3[7];
    a3[7] = 0;
    *((_QWORD *)v12 + 7) = v18;
    v12[64] = *((_BYTE *)a3 + 64);
    v19 = a3[9];
    a3[9] = 0;
    *((_QWORD *)v12 + 9) = v19;
    v20 = a3[10];
    a3[10] = 0;
    *((_QWORD *)v12 + 10) = v20;
    v21 = a3[11];
    a3[11] = 0;
    *((_QWORD *)v12 + 11) = v21;
  }
  v22 = v43;
  if ( a2 != (_QWORD *)v43 )
  {
    for ( i = v45; ; i += 96 )
    {
      if ( i )
      {
        v24 = *(_QWORD *)v22;
        *(_QWORD *)(i + 8) = 0;
        *(_QWORD *)i = v24;
        v25 = *(_QWORD *)(v22 + 8);
        *(_QWORD *)(v22 + 8) = 0;
        *(_QWORD *)(i + 8) = v25;
        LODWORD(v25) = *(_DWORD *)(v22 + 16);
        *(_QWORD *)v22 = 0;
        *(_DWORD *)(i + 16) = v25;
        *(_DWORD *)(i + 20) = *(_DWORD *)(v22 + 20);
        *(_DWORD *)(i + 24) = *(_DWORD *)(v22 + 24);
        *(_QWORD *)(i + 32) = *(_QWORD *)(v22 + 32);
        LODWORD(v25) = *(_DWORD *)(v22 + 40);
        *(_QWORD *)(v22 + 32) = 0;
        *(_DWORD *)(i + 40) = v25;
        v26 = *(_QWORD *)(v22 + 48);
        *(_QWORD *)(i + 56) = 0;
        *(_QWORD *)(i + 48) = v26;
        v27 = *(_QWORD *)(v22 + 56);
        *(_QWORD *)(v22 + 56) = 0;
        *(_QWORD *)(i + 56) = v27;
        LOBYTE(v27) = *(_BYTE *)(v22 + 64);
        *(_QWORD *)(v22 + 48) = 0;
        *(_BYTE *)(i + 64) = v27;
        *(_QWORD *)(i + 72) = *(_QWORD *)(v22 + 72);
        *(_QWORD *)(i + 80) = *(_QWORD *)(v22 + 80);
        *(_QWORD *)(i + 88) = *(_QWORD *)(v22 + 88);
        *(_QWORD *)(v22 + 88) = 0;
        *(_QWORD *)(v22 + 72) = 0;
      }
      else
      {
        v32 = *(_QWORD *)(v22 + 72);
        if ( v32 )
          j_j___libc_free_0(v32);
      }
      v28 = *(volatile signed __int32 **)(v22 + 56);
      if ( v28 )
      {
        if ( &_pthread_key_create )
        {
          v29 = _InterlockedExchangeAdd(v28 + 2, 0xFFFFFFFF);
        }
        else
        {
          v29 = *((_DWORD *)v28 + 2);
          *((_DWORD *)v28 + 2) = v29 - 1;
        }
        if ( v29 == 1 )
        {
          (*(void (**)(void))(*(_QWORD *)v28 + 16LL))();
          if ( &_pthread_key_create )
          {
            v33 = _InterlockedExchangeAdd(v28 + 3, 0xFFFFFFFF);
          }
          else
          {
            v33 = *((_DWORD *)v28 + 3);
            *((_DWORD *)v28 + 3) = v33 - 1;
          }
          if ( v33 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v28 + 24LL))(v28);
        }
      }
      v30 = *(_QWORD *)(v22 + 32);
      if ( v30 )
        j_j___libc_free_0_0(v30);
      v31 = *(volatile signed __int32 **)(v22 + 8);
      if ( v31 )
        sub_A191D0(v31);
      v22 += 96LL;
      if ( (_QWORD *)v22 == a2 )
        break;
    }
    v11 = i + 192;
  }
  if ( a2 != v4 )
  {
    v34 = a2;
    v35 = v11;
    do
    {
      v36 = *v34;
      v34 += 12;
      v35 += 96;
      *(_QWORD *)(v35 - 96) = v36;
      *(_QWORD *)(v35 - 88) = *(v34 - 11);
      *(_DWORD *)(v35 - 80) = *((_DWORD *)v34 - 20);
      *(_DWORD *)(v35 - 76) = *((_DWORD *)v34 - 19);
      *(_DWORD *)(v35 - 72) = *((_DWORD *)v34 - 18);
      *(_QWORD *)(v35 - 64) = *(v34 - 8);
      *(_DWORD *)(v35 - 56) = *((_DWORD *)v34 - 14);
      *(_QWORD *)(v35 - 48) = *(v34 - 6);
      *(_QWORD *)(v35 - 40) = *(v34 - 5);
      *(_BYTE *)(v35 - 32) = *((_BYTE *)v34 - 32);
      *(_QWORD *)(v35 - 24) = *(v34 - 3);
      *(_QWORD *)(v35 - 16) = *(v34 - 2);
      *(_QWORD *)(v35 - 8) = *(v34 - 1);
    }
    while ( v34 != v4 );
    v11 += 32
         * (3 * ((0x2AAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v34 - (char *)a2 - 96) >> 5)) & 0x7FFFFFFFFFFFFFFLL)
          + 3);
  }
  v37 = v43;
  if ( v43 )
  {
    v44 = v11;
    j_j___libc_free_0(v37);
    v11 = v44;
  }
  a1[1] = v11;
  *a1 = v45;
  a1[2] = v41;
  return v41;
}
