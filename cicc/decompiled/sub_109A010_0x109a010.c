// Function: sub_109A010
// Address: 0x109a010
//
void __fastcall sub_109A010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // edx
  int v10; // eax
  int v11; // edx
  int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r14
  unsigned __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r15
  _QWORD *v25; // rbx
  __m128i *v26; // rsi
  int v27; // r15d
  __m128i *v28; // rsi
  int v29; // r15d
  __int64 *v30; // rax
  __int64 v31; // rbx
  int v32; // eax
  __int64 v33; // r15
  _QWORD *v34; // rbx
  unsigned __int64 v35; // [rsp-60h] [rbp-60h]
  __int64 *v36; // [rsp-58h] [rbp-58h]
  __int64 *v37; // [rsp-58h] [rbp-58h]
  __int64 v38; // [rsp-50h] [rbp-50h]
  __int64 v39; // [rsp-50h] [rbp-50h]
  unsigned __int64 v40[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 == a2 )
    return;
  v8 = *(_QWORD *)a1;
  if ( a1 + 16 != *(_QWORD *)a1 && *(_QWORD *)a2 != a2 + 16 )
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v9 = *(_DWORD *)(a2 + 8);
    *(_QWORD *)a2 = v8;
    v10 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v9;
    v11 = *(_DWORD *)(a2 + 12);
    *(_DWORD *)(a2 + 8) = v10;
    v12 = *(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 12) = v11;
    *(_DWORD *)(a2 + 12) = v12;
    return;
  }
  v13 = *(unsigned int *)(a2 + 8);
  if ( v13 <= *(unsigned int *)(a1 + 12) )
  {
    v14 = *(unsigned int *)(a1 + 8);
    v15 = v14;
    if ( *(_DWORD *)(a2 + 12) >= (unsigned int)v14 )
      goto LABEL_7;
    goto LABEL_29;
  }
  v26 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v13, 0x20u, v40, a6);
  sub_1099A70(a1, v26);
  v27 = v40[0];
  if ( a1 + 16 != *(_QWORD *)a1 )
    _libc_free(*(_QWORD *)a1, v26);
  *(_DWORD *)(a1 + 12) = v27;
  v14 = *(unsigned int *)(a1 + 8);
  *(_QWORD *)a1 = v26;
  v15 = v14;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v14 )
  {
LABEL_29:
    v28 = (__m128i *)sub_C8D7D0(a2, a2 + 16, v15, 0x20u, v40, a6);
    sub_1099A70(a2, v28);
    v29 = v40[0];
    if ( a2 + 16 != *(_QWORD *)a2 )
      _libc_free(*(_QWORD *)a2, v28);
    *(_QWORD *)a2 = v28;
    *(_DWORD *)(a2 + 12) = v29;
    v14 = *(unsigned int *)(a1 + 8);
    LODWORD(v15) = *(_DWORD *)(a1 + 8);
  }
LABEL_7:
  v16 = *(unsigned int *)(a2 + 8);
  v17 = v14;
  if ( v16 <= v14 )
    v17 = *(unsigned int *)(a2 + 8);
  v35 = v17;
  if ( v17 )
  {
    v18 = 0;
    v19 = 32 * v17;
    do
    {
      v20 = v18 + *(_QWORD *)a2;
      v21 = v18 + *(_QWORD *)a1;
      v18 += 32;
      sub_22415E0(v21, v20);
    }
    while ( v19 != v18 );
    v14 = *(unsigned int *)(a1 + 8);
    v16 = *(unsigned int *)(a2 + 8);
    LODWORD(v15) = *(_DWORD *)(a1 + 8);
  }
  if ( v16 < v14 )
  {
    v22 = (__int64 *)(*(_QWORD *)a2 + 32 * v16);
    v38 = 32 * v14 + *(_QWORD *)a1;
    v23 = 32 * v35 + *(_QWORD *)a1;
    if ( v23 != v38 )
    {
      do
      {
        if ( v22 )
        {
          v36 = v22;
          *v22 = (__int64)(v22 + 2);
          sub_10990A0(v22, *(_BYTE **)v23, *(_QWORD *)v23 + *(_QWORD *)(v23 + 8));
          v22 = v36;
        }
        v23 += 32;
        v22 += 4;
      }
      while ( v38 != v23 );
      LODWORD(v15) = v14 + *(_DWORD *)(a2 + 8) - v16;
    }
    *(_DWORD *)(a2 + 8) = v15;
    v24 = *(_QWORD *)a1 + 32 * v35;
    v25 = (_QWORD *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
    while ( (_QWORD *)v24 != v25 )
    {
      while ( 1 )
      {
        v25 -= 4;
        if ( (_QWORD *)*v25 == v25 + 2 )
          break;
        j_j___libc_free_0(*v25, v25[2] + 1LL);
        if ( (_QWORD *)v24 == v25 )
          goto LABEL_25;
      }
    }
LABEL_25:
    *(_DWORD *)(a1 + 8) = v35;
  }
  else if ( v16 > v14 )
  {
    v30 = (__int64 *)(*(_QWORD *)a1 + 32 * v14);
    v39 = 32 * v16 + *(_QWORD *)a2;
    v31 = 32 * v35 + *(_QWORD *)a2;
    if ( v31 == v39 )
    {
      v32 = v14;
    }
    else
    {
      do
      {
        if ( v30 )
        {
          v37 = v30;
          *v30 = (__int64)(v30 + 2);
          sub_10990A0(v30, *(_BYTE **)v31, *(_QWORD *)v31 + *(_QWORD *)(v31 + 8));
          v30 = v37;
        }
        v31 += 32;
        v30 += 4;
      }
      while ( v39 != v31 );
      v32 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v32 + v16 - v14;
    v33 = *(_QWORD *)a2 + 32 * v35;
    v34 = (_QWORD *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
    while ( (_QWORD *)v33 != v34 )
    {
      while ( 1 )
      {
        v34 -= 4;
        if ( (_QWORD *)*v34 == v34 + 2 )
          break;
        j_j___libc_free_0(*v34, v34[2] + 1LL);
        if ( (_QWORD *)v33 == v34 )
          goto LABEL_42;
      }
    }
LABEL_42:
    *(_DWORD *)(a2 + 8) = v35;
  }
}
