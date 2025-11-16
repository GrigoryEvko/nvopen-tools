// Function: sub_3376950
// Address: 0x3376950
//
unsigned __int64 *__fastcall sub_3376950(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v8; // r14
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r10
  __int64 v12; // r15
  bool v13; // zf
  _QWORD *v14; // r10
  _QWORD *v15; // r13
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // r13
  unsigned __int64 i; // r15
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  int v26; // eax
  unsigned __int64 j; // r12
  __int64 v28; // rsi
  __int64 v29; // rsi
  unsigned __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  unsigned __int64 v36; // [rsp+10h] [rbp-50h]
  unsigned __int64 v37; // [rsp+20h] [rbp-40h]
  unsigned __int64 v38; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v38 = *a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *a1) >> 5);
  if ( v5 == 0x155555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - *a1) >> 5);
  v8 = a2;
  v9 = __CFADD__(v6, v5);
  v10 = v6 - 0x5555555555555555LL * ((__int64)(v4 - *a1) >> 5);
  v11 = a2 - v38;
  if ( v9 )
  {
    v31 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v36 = 0;
      v12 = 96;
      v37 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x155555555555555LL )
      v10 = 0x155555555555555LL;
    v31 = 96 * v10;
  }
  v33 = a3;
  v32 = sub_22077B0(v31);
  v11 = a2 - v38;
  a3 = v33;
  v37 = v32;
  v36 = v32 + v31;
  v12 = v32 + 96;
LABEL_7:
  v13 = v37 + v11 == 0;
  v14 = (_QWORD *)(v37 + v11);
  v15 = v14;
  if ( !v13 )
  {
    v16 = *(_QWORD *)(a3 + 56);
    *v14 = *(_QWORD *)a3;
    v17 = *(_QWORD *)(a3 + 8);
    v14[7] = v16;
    v14[1] = v17;
    v14[2] = *(_QWORD *)(a3 + 16);
    v14[3] = *(_QWORD *)(a3 + 24);
    v14[4] = *(_QWORD *)(a3 + 32);
    v14[5] = *(_QWORD *)(a3 + 40);
    v14[6] = *(_QWORD *)(a3 + 48);
    if ( v16 )
    {
      v34 = a3;
      sub_B96E90((__int64)(v14 + 7), v16, 1);
      a3 = v34;
    }
    v18 = *(_QWORD *)(a3 + 72);
    *((_DWORD *)v15 + 16) = *(_DWORD *)(a3 + 64);
    v15[9] = v18;
    if ( v18 )
    {
      v35 = a3;
      sub_B96E90((__int64)(v15 + 9), v18, 1);
      a3 = v35;
    }
    *((_DWORD *)v15 + 20) = *(_DWORD *)(a3 + 80);
    *((_DWORD *)v15 + 21) = *(_DWORD *)(a3 + 84);
    *((_BYTE *)v15 + 88) = *(_BYTE *)(a3 + 88);
  }
  v19 = v38;
  if ( a2 != v38 )
  {
    for ( i = v37; ; i += 96LL )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v19;
        *(_QWORD *)(i + 8) = *(_QWORD *)(v19 + 8);
        *(_QWORD *)(i + 16) = *(_QWORD *)(v19 + 16);
        *(_QWORD *)(i + 24) = *(_QWORD *)(v19 + 24);
        *(_QWORD *)(i + 32) = *(_QWORD *)(v19 + 32);
        *(_QWORD *)(i + 40) = *(_QWORD *)(v19 + 40);
        *(_QWORD *)(i + 48) = *(_QWORD *)(v19 + 48);
        v21 = *(_QWORD *)(v19 + 56);
        *(_QWORD *)(i + 56) = v21;
        if ( v21 )
          sub_B96E90(i + 56, v21, 1);
        *(_DWORD *)(i + 64) = *(_DWORD *)(v19 + 64);
        v22 = *(_QWORD *)(v19 + 72);
        *(_QWORD *)(i + 72) = v22;
        if ( v22 )
          sub_B96E90(i + 72, v22, 1);
        *(_DWORD *)(i + 80) = *(_DWORD *)(v19 + 80);
        *(_DWORD *)(i + 84) = *(_DWORD *)(v19 + 84);
        *(_BYTE *)(i + 88) = *(_BYTE *)(v19 + 88);
      }
      v19 += 96LL;
      if ( a2 == v19 )
        break;
    }
    v12 = i + 192;
  }
  if ( a2 != v4 )
  {
    do
    {
      v23 = *(_QWORD *)(v8 + 56);
      *(_QWORD *)v12 = *(_QWORD *)v8;
      v24 = *(_QWORD *)(v8 + 8);
      *(_QWORD *)(v12 + 56) = v23;
      *(_QWORD *)(v12 + 8) = v24;
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(v8 + 16);
      *(_QWORD *)(v12 + 24) = *(_QWORD *)(v8 + 24);
      *(_QWORD *)(v12 + 32) = *(_QWORD *)(v8 + 32);
      *(_QWORD *)(v12 + 40) = *(_QWORD *)(v8 + 40);
      *(_QWORD *)(v12 + 48) = *(_QWORD *)(v8 + 48);
      if ( v23 )
        sub_B96E90(v12 + 56, v23, 1);
      v25 = *(_QWORD *)(v8 + 72);
      *(_DWORD *)(v12 + 64) = *(_DWORD *)(v8 + 64);
      *(_QWORD *)(v12 + 72) = v25;
      if ( v25 )
        sub_B96E90(v12 + 72, v25, 1);
      v26 = *(_DWORD *)(v8 + 80);
      v8 += 96;
      v12 += 96;
      *(_DWORD *)(v12 - 16) = v26;
      *(_DWORD *)(v12 - 12) = *(_DWORD *)(v8 - 12);
      *(_BYTE *)(v12 - 8) = *(_BYTE *)(v8 - 8);
    }
    while ( v4 != v8 );
  }
  for ( j = v38; v4 != j; j += 96LL )
  {
    v28 = *(_QWORD *)(j + 72);
    if ( v28 )
      sub_B91220(j + 72, v28);
    v29 = *(_QWORD *)(j + 56);
    if ( v29 )
      sub_B91220(j + 56, v29);
  }
  if ( v38 )
    j_j___libc_free_0(v38);
  *a1 = v37;
  a1[1] = v12;
  a1[2] = v36;
  return a1;
}
