// Function: sub_A790C0
// Address: 0xa790c0
//
signed __int64 __fastcall sub_A790C0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  signed __int64 result; // rax
  unsigned __int64 v4; // r9
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned int v8; // ecx
  unsigned int v9; // esi
  unsigned int v10; // edi
  __int64 v11; // rax
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  _DWORD *v15; // rsi
  unsigned __int64 v16; // r13
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // rax
  unsigned int i; // edx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // r12
  unsigned int v29; // ecx
  __int64 v30; // r8
  _DWORD *v31; // [rsp-40h] [rbp-40h]

  result = a2 - a1;
  if ( (__int64)(a2 - a1) <= 256 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v18 = a2;
    goto LABEL_23;
  }
  v7 = a1 + 16;
  v31 = (_DWORD *)(a1 + 32);
  while ( 2 )
  {
    v8 = *(_DWORD *)(a1 + 16);
    v9 = *(_DWORD *)(v4 - 16);
    --v6;
    v10 = *(_DWORD *)a1;
    v11 = a1 + 16 * (result >> 5);
    v12 = *(_DWORD *)v11;
    if ( v8 >= *(_DWORD *)v11 )
    {
      if ( v8 < v9 )
        goto LABEL_7;
      if ( v12 < v9 )
      {
LABEL_17:
        *(_DWORD *)a1 = v9;
        v23 = *(_QWORD *)(v4 - 8);
        *(_DWORD *)(v4 - 16) = v10;
        v24 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 8) = v23;
        *(_QWORD *)(v4 - 8) = v24;
        v10 = *(_DWORD *)(a1 + 16);
        v8 = *(_DWORD *)a1;
        goto LABEL_8;
      }
LABEL_21:
      *(_DWORD *)a1 = v12;
      v25 = *(_QWORD *)(v11 + 8);
      *(_DWORD *)v11 = v10;
      v26 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 8) = v25;
      *(_QWORD *)(v11 + 8) = v26;
      v10 = *(_DWORD *)(a1 + 16);
      v8 = *(_DWORD *)a1;
      goto LABEL_8;
    }
    if ( v12 < v9 )
      goto LABEL_21;
    if ( v8 < v9 )
      goto LABEL_17;
LABEL_7:
    v13 = *(_QWORD *)(a1 + 8);
    v14 = *(_QWORD *)(a1 + 24);
    *(_DWORD *)a1 = v8;
    *(_DWORD *)(a1 + 16) = v10;
    *(_QWORD *)(a1 + 8) = v14;
    *(_QWORD *)(a1 + 24) = v13;
LABEL_8:
    v15 = v31;
    v16 = v7;
    v17 = v4;
    while ( 1 )
    {
      v18 = v16;
      if ( v10 < v8 )
        goto LABEL_14;
      v19 = v17 - 16;
      for ( i = *(_DWORD *)(v17 - 16); i > v8; v19 -= 16LL )
        i = *(_DWORD *)(v19 - 16);
      if ( v16 >= v19 )
        break;
      *(v15 - 4) = i;
      v21 = *(_QWORD *)(v19 + 8);
      *(_DWORD *)v19 = v10;
      v22 = *((_QWORD *)v15 - 1);
      *((_QWORD *)v15 - 1) = v21;
      *(_QWORD *)(v19 + 8) = v22;
      v8 = *(_DWORD *)a1;
      v17 = v19;
LABEL_14:
      v10 = *v15;
      v16 += 16LL;
      v15 += 4;
    }
    sub_A790C0(v16, v4, v6);
    result = v16 - a1;
    if ( (__int64)(v16 - a1) > 256 )
    {
      if ( v6 )
      {
        v4 = v16;
        continue;
      }
LABEL_23:
      v27 = result >> 4;
      v28 = ((result >> 4) - 2) >> 1;
      sub_A6DF20(a1, v28, result >> 4, *(_DWORD *)(a1 + 16 * v28), *(_QWORD *)(a1 + 16 * v28 + 8));
      do
      {
        --v28;
        sub_A6DF20(a1, v28, v27, *(_DWORD *)(a1 + 16 * v28), *(_QWORD *)(a1 + 16 * v28 + 8));
      }
      while ( v28 );
      do
      {
        v18 -= 16LL;
        v29 = *(_DWORD *)v18;
        v30 = *(_QWORD *)(v18 + 8);
        *(_DWORD *)v18 = *(_DWORD *)a1;
        *(_QWORD *)(v18 + 8) = *(_QWORD *)(a1 + 8);
        result = (signed __int64)sub_A6DF20(a1, 0, (__int64)(v18 - a1) >> 4, v29, v30);
      }
      while ( (__int64)(v18 - a1) > 16 );
    }
    return result;
  }
}
