// Function: sub_C7DB40
// Address: 0xc7db40
//
_QWORD *__fastcall sub_C7DB40(_QWORD *a1, unsigned __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  const char **v8; // rdi
  unsigned int v9; // ebx
  bool v10; // zf
  int v11; // eax
  size_t v12; // r15
  _QWORD *v13; // rsi
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  __int64 v20; // rax
  __int64 v21; // r8
  _QWORD *v22; // r9
  __int64 v23; // [rsp+0h] [rbp-160h]
  void *srca; // [rsp+8h] [rbp-158h]
  _QWORD *src; // [rsp+8h] [rbp-158h]
  _BYTE *v26; // [rsp+10h] [rbp-150h] BYREF
  size_t v27; // [rsp+18h] [rbp-148h]
  __int64 v28; // [rsp+20h] [rbp-140h]
  _BYTE v29[312]; // [rsp+28h] [rbp-138h] BYREF

  v8 = (const char **)a3;
  v9 = 4;
  v26 = v29;
  v27 = 0;
  if ( BYTE1(a4) )
    v9 = a4;
  v10 = *(_BYTE *)(a3 + 33) == 1;
  v28 = 256;
  if ( !v10 )
    goto LABEL_10;
  v11 = *(unsigned __int8 *)(a3 + 32);
  if ( (_BYTE)v11 == 1 )
  {
    v14 = 33;
    v12 = 0;
    v13 = 0;
    goto LABEL_11;
  }
  a3 = (unsigned int)(v11 - 3);
  if ( (unsigned __int8)(v11 - 3) > 3u )
  {
LABEL_10:
    sub_CA0EC0(v8, &v26);
    v12 = v27;
    v13 = v26;
    v14 = v27 + 33;
    goto LABEL_11;
  }
  if ( (_BYTE)v11 == 4 )
  {
    v12 = *((_QWORD *)*v8 + 1);
    v13 = *(_QWORD **)*v8;
    v14 = v12 + 33;
    goto LABEL_11;
  }
  if ( (unsigned __int8)v11 > 4u )
  {
    if ( (unsigned __int8)(v11 - 5) <= 1u )
    {
      v12 = (size_t)v8[1];
      v13 = *v8;
      v14 = v12 + 33;
      goto LABEL_11;
    }
LABEL_26:
    BUG();
  }
  if ( (_BYTE)v11 != 3 )
    goto LABEL_26;
  v13 = *v8;
  if ( *v8 )
  {
    v12 = strlen(*v8);
    v14 = v12 + 33;
  }
  else
  {
    v14 = 33;
    v12 = 0;
  }
LABEL_11:
  v15 = v9;
  v16 = (1LL << v9) + a2 + 1;
  v17 = 1LL << v9;
  v18 = v14 + v16;
  if ( a2 < v18 && (v23 = v14, v20 = malloc(v18, v13, a3, v15, v14, a6), v21 = v23, (v22 = (_QWORD *)v20) != 0) )
  {
    *(_QWORD *)(v20 + 24) = v12;
    if ( v12 )
    {
      src = (_QWORD *)v20;
      memcpy((void *)(v20 + 32), v13, v12);
      v21 = v23;
      v22 = src;
    }
    *((_BYTE *)v22 + v12 + 32) = 0;
    v13 = (_QWORD *)(-v17 & ((unsigned __int64)v22 + v21 + v17 - 1));
    srca = v22;
    *((_BYTE *)v13 + a2) = 0;
    *v22 = off_49DCA00;
    sub_C7DA80((__int64)v22, (__int64)v13, (__int64)v13 + a2);
    *a1 = srca;
  }
  else
  {
    *a1 = 0;
  }
  if ( v26 != v29 )
    _libc_free(v26, v13);
  return a1;
}
