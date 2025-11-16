// Function: sub_3361650
// Address: 0x3361650
//
void __fastcall sub_3361650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // r15
  unsigned int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // r8
  _QWORD *v18; // rax
  _DWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned int v22; // r9d
  __int64 v23; // [rsp+18h] [rbp-68h]
  unsigned int v24; // [rsp+2Ch] [rbp-54h] BYREF
  char v25[80]; // [rsp+30h] [rbp-50h] BYREF

  v8 = a4;
  v11 = *(_DWORD *)(a1 + 72);
  if ( !v11 )
    goto LABEL_21;
  if ( *(_QWORD *)(a6 + 88) )
  {
    v12 = *(_QWORD *)(a6 + 64);
    if ( !v12 )
      goto LABEL_10;
    v13 = a6 + 56;
    do
    {
      while ( 1 )
      {
        a4 = *(_QWORD *)(v12 + 16);
        v14 = *(_QWORD *)(v12 + 24);
        if ( v11 <= *(_DWORD *)(v12 + 32) )
          break;
        v12 = *(_QWORD *)(v12 + 24);
        if ( !v14 )
          goto LABEL_8;
      }
      v13 = v12;
      v12 = *(_QWORD *)(v12 + 16);
    }
    while ( a4 );
LABEL_8:
    if ( a6 + 56 == v13 || v11 < *(_DWORD *)(v13 + 32) )
      goto LABEL_10;
LABEL_21:
    if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
      return;
    v21 = *(_QWORD *)(a2 + 720);
    v22 = 0;
LABEL_23:
    sub_335CC30(a1, v21, a3, a5, v8, v22);
    return;
  }
  v19 = *(_DWORD **)a6;
  v20 = *(_QWORD *)a6 + 4LL * *(unsigned int *)(a6 + 8);
  if ( *(_QWORD *)a6 != v20 )
  {
    while ( v11 != *v19 )
    {
      if ( (_DWORD *)v20 == ++v19 )
        goto LABEL_10;
    }
    if ( (_DWORD *)v20 != v19 )
      goto LABEL_21;
  }
LABEL_10:
  if ( a7 )
  {
    v23 = a2;
    v24 = *(_DWORD *)(a1 + 72);
    sub_3361470((__int64)v25, a6, &v24, a4, a7);
    v15 = *(unsigned int *)(a5 + 8);
    v16 = v11;
    a2 = v23;
    v17 = a7;
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v15 + 1, 0x10u, a7, v11);
      v15 = *(unsigned int *)(a5 + 8);
      v16 = v11;
      v17 = a7;
      a2 = v23;
    }
    v18 = (_QWORD *)(*(_QWORD *)a5 + 16 * v15);
    *v18 = v16;
    v18[1] = v17;
    ++*(_DWORD *)(a5 + 8);
  }
  if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
  {
    v21 = *(_QWORD *)(a2 + 720);
    v22 = v11;
    goto LABEL_23;
  }
}
