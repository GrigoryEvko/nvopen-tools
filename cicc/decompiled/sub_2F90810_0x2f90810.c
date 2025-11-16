// Function: sub_2F90810
// Address: 0x2f90810
//
void __fastcall sub_2F90810(__int64 a1, __int64 a2, int a3, _BYTE *a4)
{
  _QWORD *v5; // rax
  __int64 v6; // rsi
  _BYTE *v7; // rsi
  _BYTE *v8; // r9
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // ecx
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r9
  int v17; // r9d
  _BYTE *v18; // rsi
  unsigned __int64 v19; // rdi
  __int64 v21; // [rsp+8h] [rbp-68h] BYREF
  unsigned __int64 v22; // [rsp+18h] [rbp-58h] BYREF
  _BYTE *v23; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v24; // [rsp+28h] [rbp-48h]
  _BYTE *v25; // [rsp+30h] [rbp-40h]

  v5 = *(_QWORD **)a1;
  v21 = a2;
  v24 = 0;
  v25 = 0;
  v6 = v5[1];
  v23 = 0;
  sub_2F8FF90((__int64)&v23, (v6 - *v5) >> 8);
  v7 = v24;
  if ( v24 == v25 )
  {
    sub_2F90070((__int64)&v23, v24, &v21);
    v8 = v24;
  }
  else
  {
    if ( v24 )
    {
      *(_QWORD *)v24 = v21;
      v7 = v24;
    }
    v8 = v7 + 8;
    v24 = v7 + 8;
  }
  while ( 1 )
  {
    v9 = *((_QWORD *)v8 - 1);
    v10 = *(_QWORD *)(a1 + 344);
    v24 = v8 - 8;
    v11 = *(_DWORD *)(v9 + 200);
    v21 = v9;
    *(_QWORD *)(v10 + 8LL * (v11 >> 6)) |= 1LL << v11;
    v12 = *(_QWORD *)(v21 + 120);
    v13 = v12 + 16LL * *(unsigned int *)(v21 + 128);
    if ( v12 != v13 )
      break;
LABEL_16:
    v8 = v24;
    if ( v24 == v23 )
    {
      if ( v24 )
        j_j___libc_free_0((unsigned __int64)v24);
      return;
    }
  }
  while ( 1 )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(a1 + 320);
      v15 = *(_QWORD *)(v13 - 16) & 0xFFFFFFFFFFFFFFF8LL;
      v16 = *(unsigned int *)(v15 + 200);
      if ( v16 < (*(_QWORD *)(a1 + 328) - v14) >> 2 )
        break;
LABEL_7:
      v13 -= 16;
      if ( v12 == v13 )
        goto LABEL_16;
    }
    v17 = *(_DWORD *)(v14 + 4 * v16);
    if ( v17 == a3 )
      break;
    if ( (*(_QWORD *)(*(_QWORD *)(a1 + 344) + 8LL * (*(_DWORD *)(v15 + 200) >> 6)) & (1LL << *(_DWORD *)(v15 + 200))) != 0
      || v17 >= a3 )
    {
      goto LABEL_7;
    }
    v22 = *(_QWORD *)(v13 - 16) & 0xFFFFFFFFFFFFFFF8LL;
    v18 = v24;
    if ( v24 == v25 )
    {
      sub_2F90680((__int64)&v23, v24, &v22);
      goto LABEL_7;
    }
    if ( v24 )
    {
      *(_QWORD *)v24 = v15;
      v18 = v24;
    }
    v13 -= 16;
    v24 = v18 + 8;
    if ( v12 == v13 )
      goto LABEL_16;
  }
  v19 = (unsigned __int64)v23;
  *a4 = 1;
  if ( v19 )
    j_j___libc_free_0(v19);
}
