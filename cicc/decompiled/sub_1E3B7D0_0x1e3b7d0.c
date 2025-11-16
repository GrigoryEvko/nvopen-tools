// Function: sub_1E3B7D0
// Address: 0x1e3b7d0
//
__int64 __fastcall sub_1E3B7D0(char **a1, char *a2, _QWORD *a3)
{
  char *v3; // rcx
  char *v4; // r9
  __int64 v5; // rax
  __int64 v7; // rdx
  char *v9; // r13
  bool v10; // cf
  unsigned __int64 v11; // rax
  signed __int64 v12; // r8
  __int64 v13; // r12
  _QWORD *v14; // r14
  _QWORD *v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // r12
  char *v18; // rbx
  _QWORD *v19; // rax
  __int64 v20; // rax
  volatile signed __int32 *v21; // rdi
  signed __int32 v22; // eax
  signed __int32 v23; // eax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  char *v30; // [rsp+18h] [rbp-48h]
  signed __int64 v31; // [rsp+18h] [rbp-48h]
  char *v32; // [rsp+20h] [rbp-40h]
  char *v33; // [rsp+20h] [rbp-40h]
  char *v34; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = (v3 - *a1) >> 4;
  if ( v5 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v5 )
    v7 = (a1[1] - *a1) >> 4;
  v9 = a2;
  v10 = __CFADD__(v7, v5);
  v11 = v7 + v5;
  v12 = a2 - v4;
  if ( v10 )
  {
    v27 = 0x7FFFFFFFFFFFFFF0LL;
LABEL_36:
    v31 = a2 - v4;
    v33 = a1[1];
    v34 = *a1;
    v28 = sub_22077B0(v27);
    v4 = v34;
    v3 = v33;
    v14 = (_QWORD *)v28;
    v12 = v31;
    v29 = v27 + v28;
    v13 = v28 + 16;
    goto LABEL_7;
  }
  if ( v11 )
  {
    if ( v11 > 0x7FFFFFFFFFFFFFFLL )
      v11 = 0x7FFFFFFFFFFFFFFLL;
    v27 = 16 * v11;
    goto LABEL_36;
  }
  v29 = 0;
  v13 = 16;
  v14 = 0;
LABEL_7:
  v15 = (_QWORD *)((char *)v14 + v12);
  if ( (_QWORD *)((char *)v14 + v12) )
  {
    *v15 = *a3;
    v16 = a3[1];
    v15[1] = v16;
    if ( v16 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd((volatile signed __int32 *)(v16 + 8), 1u);
      else
        ++*(_DWORD *)(v16 + 8);
    }
  }
  if ( a2 != v4 )
  {
    v17 = v14;
    v18 = v4;
    while ( 1 )
    {
      if ( v17 )
      {
        *v17 = *(_QWORD *)v18;
        v20 = *((_QWORD *)v18 + 1);
        *((_QWORD *)v18 + 1) = 0;
        v17[1] = v20;
        *(_QWORD *)v18 = 0;
      }
      v21 = (volatile signed __int32 *)*((_QWORD *)v18 + 1);
      if ( !v21 )
        goto LABEL_13;
      if ( &_pthread_key_create )
      {
        v22 = _InterlockedExchangeAdd(v21 + 2, 0xFFFFFFFF);
      }
      else
      {
        v22 = *((_DWORD *)v21 + 2);
        *((_DWORD *)v21 + 2) = v22 - 1;
      }
      if ( v22 == 1
        && ((v30 = v3, v32 = v4, (*(void (**)(void))(*(_QWORD *)v21 + 16LL))(), v4 = v32, v3 = v30, &_pthread_key_create)
          ? (v23 = _InterlockedExchangeAdd(v21 + 3, 0xFFFFFFFF))
          : (v23 = *((_DWORD *)v21 + 3), *((_DWORD *)v21 + 3) = v23 - 1),
            v23 == 1) )
      {
        v18 += 16;
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v21 + 24LL))(v21);
        v3 = v30;
        v19 = v17 + 2;
        v4 = v32;
        if ( v18 == a2 )
        {
LABEL_25:
          v13 = (__int64)(v17 + 4);
          break;
        }
      }
      else
      {
LABEL_13:
        v18 += 16;
        v19 = v17 + 2;
        if ( v18 == a2 )
          goto LABEL_25;
      }
      v17 = v19;
    }
  }
  if ( a2 != v3 )
  {
    v24 = v13;
    do
    {
      v25 = *(_QWORD *)v9;
      v9 += 16;
      v24 += 16;
      *(_QWORD *)(v24 - 16) = v25;
      *(_QWORD *)(v24 - 8) = *((_QWORD *)v9 - 1);
    }
    while ( v9 != v3 );
    v13 += v3 - a2;
  }
  if ( v4 )
    j_j___libc_free_0(v4, a1[2] - v4);
  *a1 = (char *)v14;
  a1[1] = (char *)v13;
  a1[2] = (char *)v29;
  return v29;
}
