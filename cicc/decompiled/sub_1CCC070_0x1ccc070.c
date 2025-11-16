// Function: sub_1CCC070
// Address: 0x1ccc070
//
void __fastcall sub_1CCC070(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // r8
  _QWORD *v9; // r14
  unsigned __int64 v10; // r12
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  __int64 v13; // rax
  __int64 *v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r12
  size_t v17; // r14
  _BYTE *v18; // rdi
  _BYTE *v19; // r9
  __int64 v20; // rbx
  __int64 v21; // rsi
  _QWORD *v22; // rdi
  _QWORD *v23; // rax
  _QWORD *v24; // r12
  __int64 v25; // rax
  __int64 v26; // rbx
  _QWORD *v27; // r12
  __int64 v28; // rsi
  _QWORD *v29; // rdi
  unsigned int v30; // [rsp-54h] [rbp-54h]
  unsigned __int64 v31; // [rsp-50h] [rbp-50h]
  _QWORD *v32; // [rsp-50h] [rbp-50h]
  _BYTE *v33; // [rsp-50h] [rbp-50h]
  size_t v34; // [rsp-40h] [rbp-40h] BYREF

  if ( (__int64 *)a1 != a2 )
  {
    v9 = *(_QWORD **)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v30 = *((_DWORD *)a2 + 2);
    v8 = v30;
    v11 = *(_QWORD **)a1;
    if ( v30 <= v10 )
    {
      v23 = *(_QWORD **)a1;
      if ( v30 )
      {
        v26 = *a2;
        v27 = &v9[4 * v30];
        do
        {
          v28 = v26;
          v29 = v9;
          v9 += 4;
          v26 += 32;
          sub_2240AE0(v29, v28);
        }
        while ( v9 != v27 );
        v23 = *(_QWORD **)a1;
        v10 = *(unsigned int *)(a1 + 8);
      }
      v24 = &v23[4 * v10];
      while ( v9 != v24 )
      {
        v24 -= 4;
        if ( (_QWORD *)*v24 != v24 + 2 )
          j_j___libc_free_0(*v24, v24[2] + 1LL);
      }
LABEL_30:
      *(_DWORD *)(a1 + 8) = v30;
      return;
    }
    if ( v30 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      if ( *(_DWORD *)(a1 + 8) )
      {
        v10 *= 32LL;
        v20 = *a2;
        v32 = (_QWORD *)((char *)v9 + v10);
        do
        {
          v21 = v20;
          v22 = v9;
          v20 += 32;
          v9 += 4;
          sub_2240AE0(v22, v21);
        }
        while ( v9 != v32 );
        v8 = *((unsigned int *)a2 + 2);
        v11 = *(_QWORD **)a1;
      }
    }
    else
    {
      v12 = &v9[4 * v10];
      while ( v12 != v9 )
      {
        while ( 1 )
        {
          v12 -= 4;
          if ( (_QWORD *)*v12 == v12 + 2 )
            break;
          v31 = v8;
          j_j___libc_free_0(*v12, v12[2] + 1LL);
          v8 = v31;
          if ( v12 == v9 )
            goto LABEL_8;
        }
      }
LABEL_8:
      *(_DWORD *)(a1 + 8) = 0;
      v10 = 0;
      sub_12BE710(a1, v8, a3, a4, v8, a6);
      v8 = *((unsigned int *)a2 + 2);
      v11 = *(_QWORD **)a1;
    }
    v13 = *a2;
    v14 = (_QWORD *)((char *)v11 + v10);
    v15 = *a2 + 32 * v8;
    v16 = v13 + v10;
    if ( v15 == v16 )
      goto LABEL_30;
    while ( 1 )
    {
      if ( !v14 )
        goto LABEL_13;
      *v14 = (__int64)(v14 + 2);
      v19 = *(_BYTE **)v16;
      v17 = *(_QWORD *)(v16 + 8);
      if ( v17 + *(_QWORD *)v16 )
      {
        if ( !v19 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
      }
      v34 = *(_QWORD *)(v16 + 8);
      if ( v17 > 0xF )
        break;
      v18 = (_BYTE *)*v14;
      if ( v17 == 1 )
      {
        *v18 = *v19;
        v17 = v34;
        v18 = (_BYTE *)*v14;
      }
      else if ( v17 )
      {
        goto LABEL_32;
      }
LABEL_12:
      v14[1] = v17;
      v18[v17] = 0;
LABEL_13:
      v16 += 32;
      v14 += 4;
      if ( v15 == v16 )
        goto LABEL_30;
    }
    v33 = v19;
    v25 = sub_22409D0(v14, &v34, 0);
    v19 = v33;
    *v14 = v25;
    v18 = (_BYTE *)v25;
    v14[2] = v34;
LABEL_32:
    memcpy(v18, v19, v17);
    v17 = v34;
    v18 = (_BYTE *)*v14;
    goto LABEL_12;
  }
}
