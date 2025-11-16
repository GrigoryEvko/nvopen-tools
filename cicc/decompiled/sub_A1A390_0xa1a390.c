// Function: sub_A1A390
// Address: 0xa1a390
//
__int64 __fastcall sub_A1A390(char **a1, char *a2, __int64 *a3)
{
  char *v3; // rcx
  char *v4; // r9
  __int64 v5; // rax
  bool v7; // zf
  __int64 v9; // rdi
  __int64 v10; // rax
  char *v11; // r13
  bool v12; // cf
  unsigned __int64 v13; // rax
  signed __int64 v14; // r8
  __int64 v15; // r12
  _QWORD *v16; // r14
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  _QWORD *v20; // r12
  char *v21; // rbx
  _QWORD *v22; // rax
  __int64 v23; // rax
  volatile signed __int32 *v24; // rdi
  signed __int32 v25; // eax
  signed __int32 v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-58h]
  char *v33; // [rsp+18h] [rbp-48h]
  signed __int64 v34; // [rsp+18h] [rbp-48h]
  char *v35; // [rsp+20h] [rbp-40h]
  char *v36; // [rsp+20h] [rbp-40h]
  char *v37; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = (v3 - *a1) >> 4;
  if ( v5 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = v5 == 0;
  v9 = (a1[1] - *a1) >> 4;
  v10 = 1;
  v11 = a2;
  if ( !v7 )
    v10 = v9;
  v12 = __CFADD__(v9, v10);
  v13 = v9 + v10;
  v14 = a2 - v4;
  if ( v12 )
  {
    v30 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v13 )
    {
      v32 = 0;
      v15 = 16;
      v16 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0x7FFFFFFFFFFFFFFLL )
      v13 = 0x7FFFFFFFFFFFFFFLL;
    v30 = 16 * v13;
  }
  v34 = a2 - v4;
  v36 = v3;
  v37 = v4;
  v31 = sub_22077B0(v30);
  v4 = v37;
  v3 = v36;
  v16 = (_QWORD *)v31;
  v14 = v34;
  v32 = v30 + v31;
  v15 = v31 + 16;
LABEL_7:
  v17 = (_QWORD *)((char *)v16 + v14);
  if ( (_QWORD *)((char *)v16 + v14) )
  {
    v18 = *a3;
    *a3 = 0;
    *v17 = v18;
    v19 = a3[1];
    a3[1] = 0;
    v17[1] = v19;
  }
  if ( a2 != v4 )
  {
    v20 = v16;
    v21 = v4;
    while ( 1 )
    {
      if ( v20 )
      {
        *v20 = *(_QWORD *)v21;
        v23 = *((_QWORD *)v21 + 1);
        *((_QWORD *)v21 + 1) = 0;
        v20[1] = v23;
        *(_QWORD *)v21 = 0;
      }
      v24 = (volatile signed __int32 *)*((_QWORD *)v21 + 1);
      if ( !v24 )
        goto LABEL_11;
      if ( &_pthread_key_create )
      {
        v25 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
      }
      else
      {
        v25 = *((_DWORD *)v24 + 2);
        *((_DWORD *)v24 + 2) = v25 - 1;
      }
      if ( v25 == 1
        && ((v33 = v3, v35 = v4, (*(void (**)(void))(*(_QWORD *)v24 + 16LL))(), v4 = v35, v3 = v33, &_pthread_key_create)
          ? (v26 = _InterlockedExchangeAdd(v24 + 3, 0xFFFFFFFF))
          : (v26 = *((_DWORD *)v24 + 3), *((_DWORD *)v24 + 3) = v26 - 1),
            v26 == 1) )
      {
        v21 += 16;
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 24LL))(v24);
        v3 = v33;
        v22 = v20 + 2;
        v4 = v35;
        if ( v21 == a2 )
        {
LABEL_23:
          v15 = (__int64)(v20 + 4);
          break;
        }
      }
      else
      {
LABEL_11:
        v21 += 16;
        v22 = v20 + 2;
        if ( v21 == a2 )
          goto LABEL_23;
      }
      v20 = v22;
    }
  }
  if ( a2 != v3 )
  {
    v27 = v15;
    do
    {
      v28 = *(_QWORD *)v11;
      v11 += 16;
      v27 += 16;
      *(_QWORD *)(v27 - 16) = v28;
      *(_QWORD *)(v27 - 8) = *((_QWORD *)v11 - 1);
    }
    while ( v11 != v3 );
    v15 += v3 - a2;
  }
  if ( v4 )
    j_j___libc_free_0(v4, a1[2] - v4);
  *a1 = (char *)v16;
  a1[1] = (char *)v15;
  a1[2] = (char *)v32;
  return v32;
}
