// Function: sub_353E330
// Address: 0x353e330
//
unsigned __int64 *__fastcall sub_353E330(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 *result; // rax
  _QWORD *v5; // r13
  __int64 v6; // rbx
  _QWORD *v7; // r10
  __int64 v8; // r12
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // r8
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // eax
  int v15; // edx
  _QWORD *v16; // rdi
  unsigned __int64 v17; // r15
  unsigned __int64 *v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned __int64 *v20; // rax
  int v21; // edx
  int v22; // eax
  int v23; // edx
  int v24; // eax
  int v25; // ecx
  int v26; // edx
  __int64 v27; // rbx
  __int64 v28; // rsi
  __int64 v29; // r12
  unsigned __int64 v30; // rcx
  int v31; // r8d
  _QWORD *v32; // r13
  unsigned __int64 v33; // rcx
  int v34; // r8d
  __int64 v35; // rbx
  _QWORD *v36; // [rsp-40h] [rbp-40h]

  result = (_QWORD *)((char *)a2 - a1);
  if ( (__int64)a2 - a1 <= 256 )
    return result;
  v5 = a2;
  v6 = a3;
  if ( !a3 )
    goto LABEL_24;
  v7 = a2;
  v8 = a1 + 16;
  v36 = (_QWORD *)(a1 + 32);
  while ( 2 )
  {
    v9 = *(_QWORD *)(a1 + 16);
    v10 = *(v7 - 2);
    --v6;
    v11 = *(_QWORD *)a1;
    v12 = a1 + 16 * ((__int64)((((__int64)v7 - a1) >> 4) + (((unsigned __int64)v7 - a1) >> 63)) >> 1);
    v13 = *(_QWORD *)v12;
    if ( v9 >= *(_QWORD *)v12 )
    {
      if ( v10 > v9 )
        goto LABEL_7;
      if ( v10 > v13 )
      {
LABEL_18:
        *(_QWORD *)a1 = v10;
        v23 = *((_DWORD *)v7 - 2);
        *(v7 - 2) = v11;
        v24 = *(_DWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 8) = v23;
        *((_DWORD *)v7 - 2) = v24;
        v11 = *(_QWORD *)(a1 + 16);
        v9 = *(_QWORD *)a1;
        goto LABEL_8;
      }
LABEL_23:
      *(_QWORD *)a1 = v13;
      v25 = *(_DWORD *)(v12 + 8);
      *(_QWORD *)v12 = v11;
      v26 = *(_DWORD *)(a1 + 8);
      *(_DWORD *)(a1 + 8) = v25;
      *(_DWORD *)(v12 + 8) = v26;
      v11 = *(_QWORD *)(a1 + 16);
      v9 = *(_QWORD *)a1;
      goto LABEL_8;
    }
    if ( v10 > v13 )
      goto LABEL_23;
    if ( v10 > v9 )
      goto LABEL_18;
LABEL_7:
    v14 = *(_DWORD *)(a1 + 8);
    v15 = *(_DWORD *)(a1 + 24);
    *(_QWORD *)a1 = v9;
    *(_QWORD *)(a1 + 16) = v11;
    *(_DWORD *)(a1 + 8) = v15;
    *(_DWORD *)(a1 + 24) = v14;
LABEL_8:
    v16 = v36;
    v17 = v8;
    v18 = v7;
    while ( 1 )
    {
      v5 = (_QWORD *)v17;
      if ( v9 > v11 )
        goto LABEL_15;
      v19 = *(v18 - 2);
      if ( v19 <= v9 )
      {
        v18 -= 2;
      }
      else
      {
        v20 = v18 - 4;
        do
        {
          v18 = v20;
          v19 = *v20;
          v20 -= 2;
        }
        while ( v19 > v9 );
      }
      if ( v17 >= (unsigned __int64)v18 )
        break;
      *(v16 - 2) = v19;
      v21 = *((_DWORD *)v18 + 2);
      *v18 = v11;
      v22 = *((_DWORD *)v16 - 2);
      *((_DWORD *)v16 - 2) = v21;
      *((_DWORD *)v18 + 2) = v22;
      v9 = *(_QWORD *)a1;
LABEL_15:
      v11 = *v16;
      v17 += 16LL;
      v16 += 2;
    }
    sub_353E330(v17, v7, v6);
    result = (unsigned __int64 *)(v17 - a1);
    if ( (__int64)(v17 - a1) > 256 )
    {
      if ( v6 )
      {
        v7 = (_QWORD *)v17;
        continue;
      }
LABEL_24:
      v27 = (__int64)result >> 4;
      v28 = (((__int64)result >> 4) - 2) >> 1;
      v29 = a1 + 16 * v28;
      while ( 1 )
      {
        v30 = *(_QWORD *)v29;
        v31 = *(_DWORD *)(v29 + 8);
        v29 -= 16;
        sub_353D500(a1, v28, v27, v30, v31);
        if ( !v28 )
          break;
        --v28;
      }
      v32 = v5 - 2;
      do
      {
        v33 = *v32;
        v34 = *((_DWORD *)v32 + 2);
        v35 = (__int64)v32 - a1;
        v32 -= 2;
        v32[2] = *(_QWORD *)a1;
        *((_DWORD *)v32 + 6) = *(_DWORD *)(a1 + 8);
        result = sub_353D500(a1, 0, v35 >> 4, v33, v34);
      }
      while ( v35 > 16 );
    }
    return result;
  }
}
