// Function: sub_205A820
// Address: 0x205a820
//
__int64 *__fastcall sub_205A820(__int64 *a1, __int64 a2, _QWORD *a3, __int64 *a4, int *a5)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // r14
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // r10
  __int64 v15; // r15
  __int64 v16; // rsi
  __int64 v17; // r10
  unsigned __int8 *v18; // rsi
  int v19; // r11d
  __int64 v20; // r10
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 v24; // rbx
  __int64 v25; // rsi
  __int64 v26; // rsi
  int v27; // eax
  __int64 i; // r13
  __int64 v29; // rsi
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-78h]
  _QWORD *v34; // [rsp+10h] [rbp-70h]
  int v36; // [rsp+18h] [rbp-68h]
  __int64 v37; // [rsp+20h] [rbp-60h]
  __int64 v38; // [rsp+20h] [rbp-60h]
  __int64 v39; // [rsp+28h] [rbp-58h]
  __int64 v40; // [rsp+38h] [rbp-48h]
  __int64 v41[7]; // [rsp+48h] [rbp-38h] BYREF

  v7 = a1[1];
  v8 = *a1;
  v9 = 0xAAAAAAAAAAAAAAABLL * ((v7 - *a1) >> 3);
  if ( v9 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  v11 = a2;
  if ( v9 )
    v10 = 0xAAAAAAAAAAAAAAABLL * ((v7 - v8) >> 3);
  v12 = __CFADD__(v10, v9);
  v13 = v10 - 0x5555555555555555LL * ((v7 - v8) >> 3);
  v14 = a2 - v8;
  if ( v12 )
  {
    v31 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v13 )
    {
      v39 = 0;
      v15 = 24;
      v40 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0x555555555555555LL )
      v13 = 0x555555555555555LL;
    v31 = 24 * v13;
  }
  v34 = a3;
  v32 = sub_22077B0(v31);
  v14 = a2 - v8;
  a3 = v34;
  v40 = v32;
  v39 = v32 + v31;
  v15 = v32 + 24;
LABEL_7:
  v16 = *a4;
  v17 = v40 + v14;
  v37 = *a3;
  v41[0] = v16;
  if ( !v16 )
  {
    if ( !v17 )
      goto LABEL_11;
    *(_QWORD *)(v17 + 8) = 0;
    v19 = *a5;
    *(_QWORD *)v17 = v37;
    goto LABEL_35;
  }
  v33 = v17;
  sub_1623A60((__int64)v41, v16, 2);
  v17 = v33;
  if ( v33 )
  {
    v18 = (unsigned __int8 *)v41[0];
    v19 = *a5;
    *(_QWORD *)(v33 + 8) = v41[0];
    *(_QWORD *)v33 = v37;
    if ( v18 )
    {
      v36 = v19;
      sub_1623210((__int64)v41, v18, v33 + 8);
      *(_DWORD *)(v33 + 16) = v36;
      goto LABEL_11;
    }
LABEL_35:
    *(_DWORD *)(v17 + 16) = v19;
    goto LABEL_11;
  }
  if ( v41[0] )
    sub_161E7C0((__int64)v41, v41[0]);
LABEL_11:
  if ( a2 != v8 )
  {
    v20 = v8;
    v38 = v8;
    v21 = v40;
    v22 = a2;
    v23 = v7;
    v24 = v20;
    while ( 1 )
    {
      if ( v21 )
      {
        *(_QWORD *)v21 = *(_QWORD *)v24;
        v25 = *(_QWORD *)(v24 + 8);
        *(_QWORD *)(v21 + 8) = v25;
        if ( v25 )
          sub_1623A60(v21 + 8, v25, 2);
        *(_DWORD *)(v21 + 16) = *(_DWORD *)(v24 + 16);
      }
      v24 += 24;
      if ( v22 == v24 )
        break;
      v21 += 24;
    }
    v7 = v23;
    a2 = v22;
    v8 = v38;
    v15 = v21 + 48;
  }
  if ( a2 != v7 )
  {
    do
    {
      v26 = *(_QWORD *)(v11 + 8);
      *(_QWORD *)v15 = *(_QWORD *)v11;
      *(_QWORD *)(v15 + 8) = v26;
      if ( v26 )
        sub_1623A60(v15 + 8, v26, 2);
      v27 = *(_DWORD *)(v11 + 16);
      v11 += 24;
      v15 += 24;
      *(_DWORD *)(v15 - 8) = v27;
    }
    while ( v7 != v11 );
  }
  for ( i = v8; i != v7; i += 24 )
  {
    v29 = *(_QWORD *)(i + 8);
    if ( v29 )
      sub_161E7C0(i + 8, v29);
  }
  if ( v8 )
    j_j___libc_free_0(v8, a1[2] - v8);
  *a1 = v40;
  a1[1] = v15;
  a1[2] = v39;
  return a1;
}
