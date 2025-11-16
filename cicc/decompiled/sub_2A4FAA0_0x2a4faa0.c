// Function: sub_2A4FAA0
// Address: 0x2a4faa0
//
signed __int64 __fastcall sub_2A4FAA0(__int64 a1, unsigned __int64 a2, __int64 a3)
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
  __int64 v28; // r13
  __int64 v29; // r12
  unsigned int v30; // ecx
  __int64 v31; // r8
  unsigned int v32; // ecx
  __int64 v33; // r8
  _DWORD *v34; // [rsp-40h] [rbp-40h]

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
  v34 = (_DWORD *)(a1 + 32);
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
    v15 = v34;
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
      if ( v19 <= v16 )
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
    sub_2A4FAA0(v16, v4, v6);
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
      v29 = a1 + 16 * v28;
      sub_2A4BA90(a1, v28, result >> 4, *(_DWORD *)v29, *(_QWORD *)(v29 + 8));
      do
      {
        v30 = *(_DWORD *)(v29 - 16);
        v31 = *(_QWORD *)(v29 - 8);
        --v28;
        v29 -= 16;
        sub_2A4BA90(a1, v28, v27, v30, v31);
      }
      while ( v28 );
      do
      {
        v18 -= 16LL;
        v32 = *(_DWORD *)v18;
        v33 = *(_QWORD *)(v18 + 8);
        *(_DWORD *)v18 = *(_DWORD *)a1;
        *(_QWORD *)(v18 + 8) = *(_QWORD *)(a1 + 8);
        result = (signed __int64)sub_2A4BA90(a1, 0, (__int64)(v18 - a1) >> 4, v32, v33);
      }
      while ( (__int64)(v18 - a1) > 16 );
    }
    return result;
  }
}
