// Function: sub_3020200
// Address: 0x3020200
//
void __fastcall sub_3020200(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // r13
  _BYTE *v9; // rax
  _BYTE *v10; // rdx
  unsigned int v11; // ebx
  __int64 v12; // r12
  char v13; // al
  char v14; // al
  int v15; // ecx
  unsigned __int64 v16; // r10
  __int64 v17; // rdx
  int v18; // r9d
  unsigned int v19; // eax
  __int64 v20; // rsi
  char v21; // r8
  unsigned int v22; // eax
  char v23; // al
  unsigned int v24; // [rsp+Ch] [rbp-74h]
  _BYTE *v25; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h]
  _BYTE v28[72]; // [rsp+38h] [rbp-48h] BYREF

  v6 = *(_DWORD *)(a2 + 8);
  v25 = v28;
  v26 = 0;
  v7 = (unsigned int)(v6 + 7) >> 3;
  v27 = 16;
  v24 = (unsigned int)(v6 + 7) >> 3;
  v8 = v7 - 1;
  if ( !(_DWORD)v7 )
    goto LABEL_8;
  v9 = v28;
  if ( v7 > 0x10 )
  {
    sub_C8D290((__int64)&v25, v28, v7, 1u, a5, a6);
    v9 = &v25[v26];
    v10 = &v25[v7];
    if ( &v25[v7] == &v25[v26] )
      goto LABEL_7;
  }
  else
  {
    v10 = &v28[v7];
    if ( &v28[v7] == v28 )
      goto LABEL_7;
  }
  do
  {
    if ( v9 )
      *v9 = 0;
    ++v9;
  }
  while ( v10 != v9 );
LABEL_7:
  v26 = v7;
  if ( v7 == 1 )
  {
    v23 = sub_C44320((unsigned __int64 *)a2, *(_DWORD *)(a2 + 8), 0);
    v15 = 1;
    *v25 = v23;
    v16 = (unsigned __int64)v25;
    v17 = *a1;
    v18 = *((_DWORD *)a1 + 2);
LABEL_11:
    v19 = *(_DWORD *)(v17 + 160);
    v20 = 0;
    do
    {
      v21 = *(_BYTE *)(v16 + v20++);
      *(_BYTE *)(*(_QWORD *)(v17 + 8) + v19) = v21;
      v19 = *(_DWORD *)(v17 + 160) + 1;
      *(_DWORD *)(v17 + 160) = v19;
    }
    while ( v15 > (int)v20 );
    goto LABEL_13;
  }
LABEL_8:
  v11 = 0;
  v12 = 0;
  do
  {
    v13 = sub_C44320((unsigned __int64 *)a2, 8, 8 * v11);
    v25[v12] = v13;
    v12 = ++v11;
  }
  while ( v11 < v8 );
  v14 = sub_C44320((unsigned __int64 *)a2, *(_DWORD *)(a2 + 8) - 8 * (int)v8, 8 * (int)v8);
  v15 = v24;
  v25[v7 - 1] = v14;
  v16 = (unsigned __int64)v25;
  v17 = *a1;
  v18 = *((_DWORD *)a1 + 2);
  if ( v24 )
    goto LABEL_11;
LABEL_13:
  if ( v18 > v15 )
  {
    v22 = *(_DWORD *)(v17 + 160);
    do
    {
      ++v15;
      *(_BYTE *)(*(_QWORD *)(v17 + 8) + v22) = 0;
      v22 = *(_DWORD *)(v17 + 160) + 1;
      *(_DWORD *)(v17 + 160) = v22;
    }
    while ( v15 != v18 );
  }
  if ( v25 != v28 )
    _libc_free((unsigned __int64)v25);
}
