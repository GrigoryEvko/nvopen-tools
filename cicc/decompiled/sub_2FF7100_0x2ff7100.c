// Function: sub_2FF7100
// Address: 0x2ff7100
//
__int64 __fastcall sub_2FF7100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // eax
  __int64 v6; // r12
  unsigned int v7; // ebx
  __int64 (__fastcall *v8)(__int64, __int64); // rax
  __int64 v9; // r13
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r8
  unsigned int v13; // r15d
  __int64 v14; // rax
  _QWORD *v15; // r13
  __int64 v16; // r9
  __int64 v17; // r15
  __int64 i; // r14
  unsigned int v19; // r12d
  __int64 v21; // rdx
  int v22; // r13d
  unsigned int v23; // r13d
  unsigned int *v24; // r13
  __int64 v25; // r12
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rbx
  unsigned int v29; // r13d
  __int64 v30; // rax
  signed int v31; // ebx
  unsigned int *v33; // [rsp+18h] [rbp-A8h]
  __int64 v34; // [rsp+18h] [rbp-A8h]
  __int64 v35; // [rsp+18h] [rbp-A8h]
  __int64 v37; // [rsp+28h] [rbp-98h]
  unsigned int v38; // [rsp+28h] [rbp-98h]
  __int64 v39; // [rsp+28h] [rbp-98h]
  unsigned int v40; // [rsp+30h] [rbp-90h]
  unsigned int v41; // [rsp+38h] [rbp-88h]
  unsigned int *v43; // [rsp+40h] [rbp-80h]
  __int64 v44; // [rsp+48h] [rbp-78h]
  __int64 v45; // [rsp+48h] [rbp-78h]
  signed int v47; // [rsp+50h] [rbp-70h]
  __int64 v48; // [rsp+58h] [rbp-68h]
  int v49; // [rsp+58h] [rbp-68h]
  unsigned int *v50; // [rsp+60h] [rbp-60h] BYREF
  __int64 v51; // [rsp+68h] [rbp-58h]
  _BYTE v52[80]; // [rsp+70h] [rbp-50h] BYREF

  v50 = (unsigned int *)v52;
  v51 = 0x800000000LL;
  v5 = *(_DWORD *)(a1 + 96);
  if ( v5 <= 1 )
  {
    return 0;
  }
  else
  {
    v48 = v5 - 1;
    v6 = 0;
    v41 = 0;
    v40 = 0;
    v44 = ~a3;
    do
    {
      v7 = v6 + 1;
      v8 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 272LL);
      if ( v8 == sub_2E85430 || a2 == ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v8)(a1, a2, v7) )
      {
        ++v6;
        v9 = 16 * v6;
        v10 = (__int64 *)(16 * v6 + *(_QWORD *)(a1 + 272));
        v11 = *v10;
        v12 = v10[1];
        if ( a3 == *v10 && a4 == v12 )
        {
          v13 = v7;
          goto LABEL_7;
        }
        if ( !(v12 & ~a4 | v11 & v44) )
        {
          v21 = (unsigned int)v51;
          if ( (unsigned __int64)(unsigned int)v51 + 1 > HIDWORD(v51) )
          {
            v35 = *v10;
            v39 = v10[1];
            sub_C8D5F0((__int64)&v50, v52, (unsigned int)v51 + 1LL, 4u, v12, v11);
            v21 = (unsigned int)v51;
            v11 = v35;
            v12 = v39;
          }
          v34 = v21;
          v37 = v11;
          v22 = sub_39FAC40(v12);
          v23 = sub_39FAC40(v37) + v22;
          v50[v34] = v7;
          LODWORD(v51) = v51 + 1;
          if ( v41 < v23 )
          {
            v41 = v23;
            v40 = v7;
          }
        }
      }
      else
      {
        ++v6;
      }
    }
    while ( v48 != v6 );
    if ( !v40 )
    {
      v19 = 0;
      v33 = v50;
      goto LABEL_17;
    }
    v13 = v40;
    v9 = 16LL * v40;
LABEL_7:
    v14 = *(unsigned int *)(a5 + 8);
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v14 + 1, 4u, v12, v11);
      v14 = *(unsigned int *)(a5 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a5 + 4 * v14) = v13;
    ++*(_DWORD *)(a5 + 8);
    v15 = (_QWORD *)(*(_QWORD *)(a1 + 272) + v9);
    v45 = *(_QWORD *)(a1 + 272);
    v16 = ~*v15;
    v17 = v16 & a3;
    for ( i = ~v15[1] & a4; ; i &= ~*(_QWORD *)(v12 + v45 + 8) )
    {
      v33 = v50;
      if ( !(v17 | i) )
      {
        v19 = 1;
        goto LABEL_17;
      }
      v43 = &v50[(unsigned int)v51];
      if ( v43 == v50 )
        break;
      v47 = 0x80000000;
      v24 = v50;
      v38 = 0;
      do
      {
        v25 = *v24;
        v26 = (__int64 *)(v45 + 16 * v25);
        v27 = *v26;
        v28 = v26[1];
        if ( v17 == *v26 && i == v28 )
        {
          v29 = *v24;
          goto LABEL_30;
        }
        v16 = v28 & ~i | v27 & ~v17;
        if ( !v16 )
        {
          v49 = sub_39FAC40(v17 & v27);
          v31 = sub_39FAC40(i & v28) + v49;
          if ( v31 > v47 )
          {
            v47 = v31;
            v38 = v25;
          }
        }
        ++v24;
      }
      while ( v43 != v24 );
      v29 = v38;
LABEL_30:
      if ( !v29 )
        break;
      v30 = *(unsigned int *)(a5 + 8);
      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        sub_C8D5F0(a5, (const void *)(a5 + 16), v30 + 1, 4u, v12, v16);
        v30 = *(unsigned int *)(a5 + 8);
      }
      v12 = 16LL * v29;
      *(_DWORD *)(*(_QWORD *)a5 + 4 * v30) = v29;
      ++*(_DWORD *)(a5 + 8);
      v45 = *(_QWORD *)(a1 + 272);
      v17 &= ~*(_QWORD *)(v12 + v45);
    }
    v19 = 0;
LABEL_17:
    if ( v33 != (unsigned int *)v52 )
      _libc_free((unsigned __int64)v33);
  }
  return v19;
}
