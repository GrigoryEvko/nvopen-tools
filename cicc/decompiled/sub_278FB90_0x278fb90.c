// Function: sub_278FB90
// Address: 0x278fb90
//
unsigned __int64 __fastcall sub_278FB90(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // r15
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rsi
  bool v13; // cf
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rbx
  unsigned __int64 v17; // rax
  int v18; // esi
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  int v23; // eax
  unsigned __int64 i; // r15
  unsigned __int64 v25; // rdi
  unsigned __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  unsigned __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int64 v35; // [rsp+20h] [rbp-40h]
  unsigned __int64 v36; // [rsp+20h] [rbp-40h]
  unsigned __int64 v37; // [rsp+20h] [rbp-40h]
  unsigned __int64 v38; // [rsp+28h] [rbp-38h]

  v6 = a2;
  v7 = a2;
  v9 = a1[1];
  v10 = *a1;
  v11 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v9 - *a1) >> 3);
  if ( v11 == 0x249249249249249LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v12 = 1;
  if ( v11 )
    v12 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v9 - v10) >> 3);
  v13 = __CFADD__(v12, v11);
  v14 = v12 + v11;
  v15 = v6 - v10;
  if ( v13 )
  {
    v27 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v14 )
    {
      v31 = 0;
      v16 = 56;
      v38 = 0;
      goto LABEL_7;
    }
    if ( v14 > 0x249249249249249LL )
      v14 = 0x249249249249249LL;
    v27 = 56 * v14;
  }
  v29 = a3;
  v33 = v6;
  v36 = v6 - v10;
  v28 = sub_22077B0(v27);
  v15 = v36;
  v6 = v33;
  v38 = v28;
  a3 = v29;
  v31 = v28 + v27;
  v16 = v28 + 56;
LABEL_7:
  v17 = v38 + v15;
  if ( v38 + v15 )
  {
    v18 = *(_DWORD *)a3;
    *(_QWORD *)(v17 + 24) = 0x400000000LL;
    a4 = *(unsigned int *)(a3 + 24);
    *(_DWORD *)v17 = v18;
    *(_BYTE *)(v17 + 4) = *(_BYTE *)(a3 + 4);
    *(_QWORD *)(v17 + 8) = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(v17 + 16) = v17 + 32;
    if ( (_DWORD)a4 )
    {
      v30 = v6;
      v34 = a3;
      v37 = v17;
      sub_2789770(v17 + 16, a3 + 16, a3, a4, v6, a6);
      v6 = v30;
      a3 = v34;
      v17 = v37;
    }
    *(_QWORD *)(v17 + 48) = *(_QWORD *)(a3 + 48);
  }
  if ( v6 != v10 )
  {
    v19 = v38;
    v20 = v10;
    while ( 1 )
    {
      if ( v19 )
      {
        *(_DWORD *)v19 = *(_DWORD *)v20;
        *(_BYTE *)(v19 + 4) = *(_BYTE *)(v20 + 4);
        v21 = *(_QWORD *)(v20 + 8);
        *(_DWORD *)(v19 + 24) = 0;
        *(_QWORD *)(v19 + 8) = v21;
        *(_QWORD *)(v19 + 16) = v19 + 32;
        *(_DWORD *)(v19 + 28) = 4;
        a3 = *(unsigned int *)(v20 + 24);
        if ( (_DWORD)a3 )
        {
          v32 = v6;
          v35 = v20;
          sub_2789770(v19 + 16, v20 + 16, a3, a4, v6, a6);
          v6 = v32;
          v20 = v35;
        }
        *(_QWORD *)(v19 + 48) = *(_QWORD *)(v20 + 48);
      }
      v20 += 56LL;
      if ( v6 == v20 )
        break;
      v19 += 56LL;
    }
    v16 = v19 + 112;
  }
  if ( v6 != v9 )
  {
    do
    {
      v23 = *(_DWORD *)v7;
      *(_DWORD *)(v16 + 24) = 0;
      *(_DWORD *)(v16 + 28) = 4;
      *(_DWORD *)v16 = v23;
      *(_BYTE *)(v16 + 4) = *(_BYTE *)(v7 + 4);
      *(_QWORD *)(v16 + 8) = *(_QWORD *)(v7 + 8);
      *(_QWORD *)(v16 + 16) = v16 + 32;
      if ( *(_DWORD *)(v7 + 24) )
        sub_2789770(v16 + 16, v7 + 16, a3, a4, v6, a6);
      v22 = *(_QWORD *)(v7 + 48);
      v7 += 56;
      v16 += 56;
      *(_QWORD *)(v16 - 8) = v22;
    }
    while ( v9 != v7 );
  }
  for ( i = v10; v9 != i; i += 56LL )
  {
    v25 = *(_QWORD *)(i + 16);
    if ( v25 != i + 32 )
      _libc_free(v25);
  }
  if ( v10 )
    j_j___libc_free_0(v10);
  a1[1] = v16;
  *a1 = v38;
  a1[2] = v31;
  return v31;
}
