// Function: sub_1612750
// Address: 0x1612750
//
__int64 __fastcall sub_1612750(_DWORD *a1, int a2, __int64 a3, void *a4, size_t a5, __int64 a6, int **a7, size_t a8)
{
  _DWORD *v8; // r12
  size_t v9; // rdx
  int **v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 result; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // rcx
  int v20; // ebx
  __int64 v21; // rax
  _DWORD *v22; // r8
  __int64 v23; // rcx
  __int64 v24; // rax
  _QWORD *v25; // rsi
  unsigned __int64 v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  int v31; // edi
  __int64 v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rdx
  int v35; // eax
  __int64 v37; // [rsp+8h] [rbp-88h]
  void *s2; // [rsp+10h] [rbp-80h] BYREF
  size_t n; // [rsp+18h] [rbp-78h]
  _QWORD v40[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v41; // [rsp+30h] [rbp-60h]
  int *v42[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v43; // [rsp+50h] [rbp-40h]

  v8 = a1;
  v9 = a8;
  v10 = a7;
  if ( *(_QWORD *)(*((_QWORD *)a1 + 27) + 32LL) )
  {
    s2 = a7;
    n = a8;
  }
  else
  {
    s2 = a4;
    n = a5;
  }
  v11 = (unsigned int)a1[58];
  if ( a1[58] )
  {
    v12 = 0;
    v13 = *((_QWORD *)a1 + 28);
    while ( 1 )
    {
      if ( *(_QWORD *)(v13 + 8) == n )
      {
        v37 = v11;
        if ( !n )
          break;
        a1 = *(_DWORD **)v13;
        v10 = (int **)s2;
        v35 = memcmp(*(const void **)v13, s2, n);
        v11 = v37;
        if ( !v35 )
          break;
      }
      ++v12;
      v13 += 48;
      if ( v11 == v12 )
        goto LABEL_7;
    }
  }
  else
  {
LABEL_7:
    v14 = sub_16E8CB0(a1, v10, v9);
    v10 = v42;
    v43 = 770;
    a1 = v8;
    v41 = 1283;
    v40[0] = "Cannot find option named '";
    v40[1] = &s2;
    v42[0] = (int *)v40;
    v42[1] = (int *)"'!";
    result = sub_16B1F90(v8, v42, 0, 0, v14);
    if ( (_BYTE)result )
      return result;
  }
  --*((_QWORD *)v8 + 21);
  *((_QWORD *)v8 + 24) -= 4LL;
  v8[4] = a2;
  v17 = sub_16D5D50(a1, v10, v9);
  v18 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    a1 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v19 = v18[2];
        v16 = v18[3];
        if ( v17 <= v18[4] )
          break;
        v18 = (_QWORD *)v18[3];
        if ( !v16 )
          goto LABEL_13;
      }
      a1 = v18;
      v18 = (_QWORD *)v18[2];
    }
    while ( v19 );
LABEL_13:
    v20 = -1;
    if ( a1 != dword_4FA0208 && v17 >= *((_QWORD *)a1 + 4) )
    {
      v21 = *((_QWORD *)a1 + 7);
      v22 = a1 + 12;
      if ( v21 )
      {
        v17 = (unsigned int)v8[2];
        a1 += 12;
        do
        {
          while ( 1 )
          {
            v23 = *(_QWORD *)(v21 + 16);
            v16 = *(_QWORD *)(v21 + 24);
            if ( *(_DWORD *)(v21 + 32) >= (int)v17 )
              break;
            v21 = *(_QWORD *)(v21 + 24);
            if ( !v16 )
              goto LABEL_20;
          }
          a1 = (_DWORD *)v21;
          v21 = *(_QWORD *)(v21 + 16);
        }
        while ( v23 );
LABEL_20:
        v20 = -1;
        if ( v22 != a1 && (int)v17 >= a1[8] )
          v20 = a1[9] - 1;
      }
    }
  }
  else
  {
    v20 = -1;
  }
  v24 = sub_16D5D50(a1, v17, v16);
  v25 = dword_4FA0208;
  v40[0] = v24;
  v26 = v24;
  v27 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_30;
  do
  {
    while ( 1 )
    {
      v28 = v27[2];
      v29 = v27[3];
      if ( v26 <= v27[4] )
        break;
      v27 = (_QWORD *)v27[3];
      if ( !v29 )
        goto LABEL_28;
    }
    v25 = v27;
    v27 = (_QWORD *)v27[2];
  }
  while ( v28 );
LABEL_28:
  if ( v25 == (_QWORD *)dword_4FA0208 || v26 < v25[4] )
  {
LABEL_30:
    v42[0] = (int *)v40;
    v25 = (_QWORD *)sub_1612390(&qword_4FA0200, v25, (unsigned __int64 **)v42);
  }
  v30 = v25[7];
  if ( v30 )
  {
    v31 = v8[2];
    v32 = (__int64)(v25 + 6);
    do
    {
      while ( 1 )
      {
        v33 = *(_QWORD *)(v30 + 16);
        v34 = *(_QWORD *)(v30 + 24);
        if ( *(_DWORD *)(v30 + 32) >= v31 )
          break;
        v30 = *(_QWORD *)(v30 + 24);
        if ( !v34 )
          goto LABEL_36;
      }
      v32 = v30;
      v30 = *(_QWORD *)(v30 + 16);
    }
    while ( v33 );
LABEL_36:
    if ( v25 + 6 != (_QWORD *)v32 && v31 >= *(_DWORD *)(v32 + 32) )
      goto LABEL_39;
  }
  else
  {
    v32 = (__int64)(v25 + 6);
  }
  v42[0] = v8 + 2;
  v32 = sub_1612460(v25 + 5, v32, v42);
LABEL_39:
  *(_DWORD *)(v32 + 36) = v20;
  return 0;
}
