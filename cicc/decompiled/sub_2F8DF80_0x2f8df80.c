// Function: sub_2F8DF80
// Address: 0x2f8df80
//
unsigned __int8 *__fastcall sub_2F8DF80(__int64 a1, __int64 a2, int a3, unsigned __int8 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdx
  __int64 v11; // rdx
  char *v12; // r15
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rsi
  __int64 v15; // r8
  int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // r14
  _BYTE *v19; // rdi
  unsigned int v20; // esi
  int v21; // r15d
  __int64 v22; // rdi
  _QWORD *v23; // r10
  unsigned int v24; // ecx
  _QWORD *v25; // rax
  __int64 v26; // rdx
  unsigned __int8 *result; // rax
  int v28; // eax
  int v29; // edx
  __int64 v30; // rdi
  char *v31; // r15
  int v32; // eax
  int v33; // esi
  __int64 v34; // rdi
  unsigned int v35; // ecx
  __int64 v36; // rax
  int v37; // r9d
  _QWORD *v38; // r8
  int v39; // eax
  int v40; // ecx
  __int64 v41; // rsi
  _QWORD *v42; // rdi
  unsigned int v43; // r14d
  int v44; // r8d
  __int64 v45; // rax
  __int64 v46; // [rsp+10h] [rbp-90h] BYREF
  int v47; // [rsp+18h] [rbp-88h]
  unsigned __int8 v48; // [rsp+1Ch] [rbp-84h]
  _QWORD v49[2]; // [rsp+20h] [rbp-80h] BYREF
  _BYTE v50[48]; // [rsp+30h] [rbp-70h] BYREF
  int v51; // [rsp+60h] [rbp-40h]

  v47 = a3;
  v10 = *(unsigned int *)(a5 + 8);
  v48 = a4;
  v46 = a2;
  v49[0] = v50;
  v49[1] = 0x600000000LL;
  if ( (_DWORD)v10 )
    sub_2F8AAD0((__int64)v49, a5, v10, (__int64)v50, a5, a6);
  v11 = *(unsigned int *)(a1 + 1312);
  v12 = (char *)&v46;
  v13 = *(unsigned int *)(a1 + 1316);
  v14 = *(_QWORD *)(a1 + 1304);
  v15 = v11 + 1;
  v51 = *(_DWORD *)(a5 + 64);
  v16 = v11;
  if ( v11 + 1 > v13 )
  {
    v30 = a1 + 1304;
    if ( v14 > (unsigned __int64)&v46 || (unsigned __int64)&v46 >= v14 + 88 * v11 )
    {
      sub_2F8CBB0(v30, v11 + 1, v11, (__int64)v50, v15, a6);
      v11 = *(unsigned int *)(a1 + 1312);
      v14 = *(_QWORD *)(a1 + 1304);
      v16 = *(_DWORD *)(a1 + 1312);
    }
    else
    {
      v31 = (char *)&v46 - v14;
      sub_2F8CBB0(v30, v11 + 1, v11, (__int64)v50, v15, a6);
      v14 = *(_QWORD *)(a1 + 1304);
      v11 = *(unsigned int *)(a1 + 1312);
      v12 = &v31[v14];
      v16 = *(_DWORD *)(a1 + 1312);
    }
  }
  v17 = 11 * v11;
  v18 = v14 + 8 * v17;
  if ( v18 )
  {
    *(_QWORD *)v18 = *(_QWORD *)v12;
    *(_DWORD *)(v18 + 8) = *((_DWORD *)v12 + 2);
    *(_BYTE *)(v18 + 12) = v12[12];
    *(_QWORD *)(v18 + 16) = v18 + 32;
    *(_QWORD *)(v18 + 24) = 0x600000000LL;
    if ( *((_DWORD *)v12 + 6) )
      sub_2F8ABB0(v18 + 16, (char **)v12 + 2, v17, (__int64)v50, v15, a6);
    *(_DWORD *)(v18 + 80) = *((_DWORD *)v12 + 20);
    v16 = *(_DWORD *)(a1 + 1312);
  }
  v19 = (_BYTE *)v49[0];
  *(_DWORD *)(a1 + 1312) = v16 + 1;
  if ( v19 != v50 )
    _libc_free((unsigned __int64)v19);
  v20 = *(_DWORD *)(a1 + 2080);
  if ( !v20 )
  {
    ++*(_QWORD *)(a1 + 2056);
    goto LABEL_34;
  }
  v21 = 1;
  v22 = *(_QWORD *)(a1 + 2064);
  v23 = 0;
  v24 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v25 = (_QWORD *)(v22 + 16LL * v24);
  v26 = *v25;
  if ( *v25 == a2 )
  {
LABEL_12:
    result = (unsigned __int8 *)(v25 + 1);
    goto LABEL_13;
  }
  while ( v26 != -4096 )
  {
    if ( !v23 && v26 == -8192 )
      v23 = v25;
    v24 = (v20 - 1) & (v21 + v24);
    v25 = (_QWORD *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( *v25 == a2 )
      goto LABEL_12;
    ++v21;
  }
  if ( !v23 )
    v23 = v25;
  v28 = *(_DWORD *)(a1 + 2072);
  ++*(_QWORD *)(a1 + 2056);
  v29 = v28 + 1;
  if ( 4 * (v28 + 1) >= 3 * v20 )
  {
LABEL_34:
    sub_2F84800(a1 + 2056, 2 * v20);
    v32 = *(_DWORD *)(a1 + 2080);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 2064);
      v35 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v29 = *(_DWORD *)(a1 + 2072) + 1;
      v23 = (_QWORD *)(v34 + 16LL * v35);
      v36 = *v23;
      if ( *v23 != a2 )
      {
        v37 = 1;
        v38 = 0;
        while ( v36 != -4096 )
        {
          if ( !v38 && v36 == -8192 )
            v38 = v23;
          v35 = v33 & (v37 + v35);
          v23 = (_QWORD *)(v34 + 16LL * v35);
          v36 = *v23;
          if ( *v23 == a2 )
            goto LABEL_26;
          ++v37;
        }
        if ( v38 )
          v23 = v38;
      }
      goto LABEL_26;
    }
    goto LABEL_57;
  }
  if ( v20 - *(_DWORD *)(a1 + 2076) - v29 <= v20 >> 3 )
  {
    sub_2F84800(a1 + 2056, v20);
    v39 = *(_DWORD *)(a1 + 2080);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 2064);
      v42 = 0;
      v43 = (v39 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v44 = 1;
      v29 = *(_DWORD *)(a1 + 2072) + 1;
      v23 = (_QWORD *)(v41 + 16LL * v43);
      v45 = *v23;
      if ( *v23 != a2 )
      {
        while ( v45 != -4096 )
        {
          if ( v45 == -8192 && !v42 )
            v42 = v23;
          v43 = v40 & (v44 + v43);
          v23 = (_QWORD *)(v41 + 16LL * v43);
          v45 = *v23;
          if ( *v23 == a2 )
            goto LABEL_26;
          ++v44;
        }
        if ( v42 )
          v23 = v42;
      }
      goto LABEL_26;
    }
LABEL_57:
    ++*(_DWORD *)(a1 + 2072);
    BUG();
  }
LABEL_26:
  *(_DWORD *)(a1 + 2072) = v29;
  if ( *v23 != -4096 )
    --*(_DWORD *)(a1 + 2076);
  *v23 = a2;
  result = (unsigned __int8 *)(v23 + 1);
  *((_BYTE *)v23 + 8) = 0;
LABEL_13:
  *result = a4;
  if ( *(_BYTE *)a1 < a4 )
    *(_BYTE *)a1 = a4;
  return result;
}
