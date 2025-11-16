// Function: sub_BC6650
// Address: 0xbc6650
//
__int64 __fastcall sub_BC6650(_QWORD *a1, int **a2, __int64 a3, int **a4, size_t a5, int a6, int **s2, size_t n)
{
  _QWORD *v8; // r13
  int **v9; // r14
  __int64 v10; // rcx
  size_t v11; // r15
  __int64 v12; // rbx
  __int64 v13; // r12
  int v14; // eax
  __int64 v15; // r12
  _QWORD *v16; // rbx
  unsigned __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  _QWORD *v22; // r8
  __int64 v23; // rcx
  int v24; // ebx
  _QWORD *v25; // r12
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  int v31; // edi
  __int64 v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 result; // rax
  __int64 v36; // rax
  __int16 v37; // [rsp+4h] [rbp-9Ch]
  __int64 v38; // [rsp+8h] [rbp-98h]
  _QWORD v39[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v40; // [rsp+30h] [rbp-70h]
  int *v41[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v42; // [rsp+60h] [rbp-40h]

  v8 = a1;
  v37 = (__int16)a2;
  if ( *(_QWORD *)(a1[28] + 32LL) )
  {
    a4 = s2;
    a5 = n;
  }
  v9 = a4;
  v10 = *((unsigned int *)a1 + 60);
  v11 = a5;
  if ( *((_DWORD *)a1 + 60) )
  {
    v12 = a1[29];
    v13 = 0;
    while ( 1 )
    {
      if ( v11 == *(_QWORD *)(v12 + 8) )
      {
        v38 = v10;
        if ( !v11 )
          break;
        a1 = *(_QWORD **)v12;
        a2 = v9;
        v14 = memcmp(*(const void **)v12, v9, v11);
        v10 = v38;
        if ( !v14 )
          break;
      }
      ++v13;
      v12 += 48;
      if ( v10 == v13 )
        goto LABEL_43;
    }
LABEL_9:
    --v8[18];
    v8[25] -= 4LL;
    *((_WORD *)v8 + 7) = v37;
    v15 = sub_C52410(a1, a2, a3);
    v16 = (_QWORD *)(v15 + 8);
    v17 = sub_C959E0();
    v18 = *(_QWORD **)(v15 + 16);
    if ( v18 )
    {
      a1 = (_QWORD *)(v15 + 8);
      do
      {
        while ( 1 )
        {
          a2 = (int **)v18[2];
          v19 = v18[3];
          if ( v17 <= v18[4] )
            break;
          v18 = (_QWORD *)v18[3];
          if ( !v19 )
            goto LABEL_14;
        }
        a1 = v18;
        v18 = (_QWORD *)v18[2];
      }
      while ( a2 );
LABEL_14:
      if ( v16 != a1 && v17 >= a1[4] )
        v16 = a1;
    }
    if ( v16 == (_QWORD *)(sub_C52410(a1, a2, v17) + 8) || (v21 = v16[7], v22 = v16 + 6, !v21) )
    {
      v24 = -1;
    }
    else
    {
      a2 = (int **)*((unsigned int *)v8 + 2);
      a1 = v16 + 6;
      do
      {
        while ( 1 )
        {
          v23 = *(_QWORD *)(v21 + 16);
          v20 = *(_QWORD *)(v21 + 24);
          if ( *(_DWORD *)(v21 + 32) >= (int)a2 )
            break;
          v21 = *(_QWORD *)(v21 + 24);
          if ( !v20 )
            goto LABEL_23;
        }
        a1 = (_QWORD *)v21;
        v21 = *(_QWORD *)(v21 + 16);
      }
      while ( v23 );
LABEL_23:
      v24 = -1;
      if ( v22 != a1 && (int)a2 >= *((_DWORD *)a1 + 8) )
        v24 = *((_DWORD *)a1 + 9) - 1;
    }
    v25 = (_QWORD *)sub_C52410(a1, a2, v20);
    v39[0] = sub_C959E0();
    v26 = (_QWORD *)v25[2];
    v27 = v25 + 1;
    if ( !v26 )
      goto LABEL_33;
    do
    {
      while ( 1 )
      {
        v28 = v26[2];
        v29 = v26[3];
        if ( v39[0] <= v26[4] )
          break;
        v26 = (_QWORD *)v26[3];
        if ( !v29 )
          goto LABEL_31;
      }
      v27 = v26;
      v26 = (_QWORD *)v26[2];
    }
    while ( v28 );
LABEL_31:
    if ( v25 + 1 == v27 || v39[0] < v27[4] )
    {
LABEL_33:
      v41[0] = (int *)v39;
      v27 = (_QWORD *)sub_BC64D0(v25, v27, (unsigned __int64 **)v41);
    }
    v30 = v27[7];
    if ( v30 )
    {
      v31 = *((_DWORD *)v8 + 2);
      v32 = (__int64)(v27 + 6);
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
            goto LABEL_39;
        }
        v32 = v30;
        v30 = *(_QWORD *)(v30 + 16);
      }
      while ( v33 );
LABEL_39:
      if ( v27 + 6 != (_QWORD *)v32 && v31 >= *(_DWORD *)(v32 + 32) )
        goto LABEL_42;
    }
    else
    {
      v32 = (__int64)(v27 + 6);
    }
    v41[0] = (int *)(v8 + 1);
    v32 = sub_BC65A0(v27 + 5, v32, v41);
LABEL_42:
    *(_DWORD *)(v32 + 36) = v24;
    return 0;
  }
LABEL_43:
  v36 = sub_CEADF0();
  a2 = v41;
  v42 = 770;
  a1 = v8;
  v40 = 1283;
  v39[0] = "Cannot find option named '";
  v41[0] = (int *)v39;
  v39[2] = v9;
  v39[3] = v11;
  v41[2] = (int *)"'!";
  result = sub_C53280(v8, v41, 0, 0, v36);
  if ( !(_BYTE)result )
    goto LABEL_9;
  return result;
}
