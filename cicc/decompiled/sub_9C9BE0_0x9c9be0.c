// Function: sub_9C9BE0
// Address: 0x9c9be0
//
__int64 __fastcall sub_9C9BE0(__int64 *a1, __int64 a2, _QWORD *a3, _DWORD *a4)
{
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r15
  bool v9; // zf
  __int64 v11; // rdi
  __int64 v12; // rax
  bool v13; // cf
  unsigned __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rbx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  int v23; // eax
  __int64 i; // r15
  __int64 v25; // rdx
  __int64 v27; // rbx
  __int64 v28; // rax
  _DWORD *v29; // [rsp+0h] [rbp-60h]
  _DWORD *v30; // [rsp+8h] [rbp-58h]
  _QWORD *v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  __int64 v37; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = (v4 - *a1) >> 5;
  if ( v6 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = a2;
  v8 = a2;
  v9 = v6 == 0;
  v11 = (a1[1] - *a1) >> 5;
  v12 = 1;
  if ( !v9 )
    v12 = v11;
  v13 = __CFADD__(v11, v12);
  v14 = v11 + v12;
  v15 = a2 - v5;
  if ( v13 )
  {
    v27 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v14 )
    {
      v32 = 0;
      v16 = 32;
      v37 = 0;
      goto LABEL_7;
    }
    if ( v14 > 0x3FFFFFFFFFFFFFFLL )
      v14 = 0x3FFFFFFFFFFFFFFLL;
    v27 = 32 * v14;
  }
  v29 = a4;
  v31 = a3;
  v35 = v7;
  v28 = sub_22077B0(v27);
  v7 = v35;
  a3 = v31;
  a4 = v29;
  v37 = v28;
  v32 = v28 + v27;
  v16 = v28 + 32;
LABEL_7:
  v17 = v37 + v15;
  if ( v17 )
  {
    v18 = *a3;
    *(_QWORD *)v17 = 6;
    *(_QWORD *)(v17 + 8) = 0;
    *(_QWORD *)(v17 + 16) = v18;
    if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
    {
      v30 = a4;
      v33 = v7;
      sub_BD73F0(v17);
      a4 = v30;
      v7 = v33;
    }
    *(_DWORD *)(v17 + 24) = *a4;
  }
  if ( v7 != v5 )
  {
    v19 = v37;
    v20 = v5;
    while ( 1 )
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = 6;
        *(_QWORD *)(v19 + 8) = 0;
        v21 = *(_QWORD *)(v20 + 16);
        *(_QWORD *)(v19 + 16) = v21;
        if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
        {
          v34 = v7;
          v36 = v20;
          sub_BD6050(v19, *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL);
          v7 = v34;
          v20 = v36;
        }
        *(_DWORD *)(v19 + 24) = *(_DWORD *)(v20 + 24);
      }
      v20 += 32;
      if ( v7 == v20 )
        break;
      v19 += 32;
    }
    v16 = v19 + 64;
  }
  if ( v7 != v4 )
  {
    do
    {
      v22 = *(_QWORD *)(v8 + 16);
      *(_QWORD *)v16 = 6;
      *(_QWORD *)(v16 + 8) = 0;
      *(_QWORD *)(v16 + 16) = v22;
      if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
        sub_BD6050(v16, *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL);
      v23 = *(_DWORD *)(v8 + 24);
      v8 += 32;
      v16 += 32;
      *(_DWORD *)(v16 - 8) = v23;
    }
    while ( v4 != v8 );
  }
  for ( i = v5; i != v4; i += 32 )
  {
    v25 = *(_QWORD *)(i + 16);
    if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
      sub_BD60C0(i);
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  a1[1] = v16;
  *a1 = v37;
  a1[2] = v32;
  return v32;
}
