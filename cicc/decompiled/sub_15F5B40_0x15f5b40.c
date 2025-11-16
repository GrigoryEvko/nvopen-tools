// Function: sub_15F5B40
// Address: 0x15f5b40
//
__int64 __fastcall sub_15F5B40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v8; // r12
  __int64 *v9; // rbx
  bool v10; // zf
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r9
  unsigned __int64 v19; // rdi
  __int64 v20; // rdi
  int v21; // r13d
  __int64 v22; // r14
  __int64 v23; // r8
  _QWORD *v24; // r13
  __int64 *v25; // rcx
  __int64 v26; // r11
  _QWORD *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // r8
  unsigned __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // r15
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  int v37; // r12d
  __int64 v38; // r14
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v43; // [rsp+8h] [rbp-48h]
  int v45; // [rsp+18h] [rbp-38h]
  __int64 v46; // [rsp+18h] [rbp-38h]

  v8 = a1;
  v9 = a7;
  v10 = *(_QWORD *)(a1 - 24) == 0;
  *(_QWORD *)(a1 + 64) = a2;
  if ( !v10 )
  {
    v11 = *(_QWORD *)(a1 - 16);
    v12 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *(_QWORD *)(a1 - 24) = a3;
  if ( a3 )
  {
    v13 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 16) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = (a1 - 16) | *(_QWORD *)(v13 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a3 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a3 + 8) = a1 - 24;
  }
  v14 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v15 = (_QWORD *)(a1 - 24 * v14);
  v16 = (8 * a5) >> 3;
  if ( 8 * a5 > 0 )
  {
    do
    {
      v17 = *a4;
      if ( *v15 )
      {
        v18 = v15[1];
        v19 = v15[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v19 = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
      }
      *v15 = v17;
      if ( v17 )
      {
        v20 = *(_QWORD *)(v17 + 8);
        v15[1] = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = (unsigned __int64)(v15 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
        v15[2] = (v17 + 8) | v15[2] & 3LL;
        *(_QWORD *)(v17 + 8) = v15;
      }
      ++a4;
      v15 += 3;
      --v16;
    }
    while ( v16 );
    v14 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
  }
  v21 = a5;
  v22 = (__int64)a7;
  v23 = (unsigned int)a5 - v14;
  if ( &a7[7 * a8] != a7 )
  {
    v45 = v21;
    v24 = (_QWORD *)(v8 + 24 * v23);
    do
    {
      v25 = *(__int64 **)(v22 + 32);
      v26 = *(_QWORD *)(v22 + 40) - (_QWORD)v25;
      if ( v26 > 0 )
      {
        v27 = v24;
        v28 = (__int64)(*(_QWORD *)(v22 + 40) - (_QWORD)v25) >> 3;
        do
        {
          v29 = *v25;
          if ( *v27 )
          {
            v30 = v27[1];
            v31 = v27[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v31 = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = *(_QWORD *)(v30 + 16) & 3LL | v31;
          }
          *v27 = v29;
          if ( v29 )
          {
            v32 = *(_QWORD *)(v29 + 8);
            v27[1] = v32;
            if ( v32 )
              *(_QWORD *)(v32 + 16) = (unsigned __int64)(v27 + 1) | *(_QWORD *)(v32 + 16) & 3LL;
            v27[2] = (v29 + 8) | v27[2] & 3LL;
            *(_QWORD *)(v29 + 8) = v27;
          }
          ++v25;
          v27 += 3;
          --v28;
        }
        while ( v28 );
        v24 += 3 * (v26 >> 3);
      }
      v22 += 56;
    }
    while ( &a7[7 * a8] != (__int64 *)v22 );
    v21 = v45;
    v9 = a7;
  }
  v33 = *(_QWORD *)sub_16498A0(v8);
  if ( *(char *)(v8 + 23) < 0 )
  {
    v34 = sub_1648A40(v8);
    v46 = v35 + v34;
    if ( *(char *)(v8 + 23) >= 0 )
      v36 = 0;
    else
      v36 = sub_1648A40(v8);
    if ( v46 != v36 )
    {
      v43 = v8;
      v37 = v21;
      v38 = v36;
      do
      {
        v39 = v9[1];
        v40 = *v9;
        v38 += 16;
        v9 += 7;
        v41 = sub_16052C0(v33, v40, v39);
        *(_DWORD *)(v38 - 8) = v37;
        *(_QWORD *)(v38 - 16) = v41;
        v37 += (*(v9 - 2) - *(v9 - 3)) >> 3;
        *(_DWORD *)(v38 - 4) = v37;
      }
      while ( v46 != v38 );
      v8 = v43;
    }
  }
  return sub_164B780(v8, a6);
}
