// Function: sub_C7D7A0
// Address: 0xc7d7a0
//
__int64 __fastcall sub_C7D7A0(__int64 a1, const char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const char **v7; // rdi
  bool v8; // zf
  int v9; // eax
  const char **v10; // r14
  size_t v11; // r15
  __int64 v12; // rax
  __int64 v13; // r12
  _BYTE *v14; // rcx
  const char **v15; // rdi
  const char **v17; // [rsp+0h] [rbp-150h] BYREF
  size_t v18; // [rsp+8h] [rbp-148h]
  __int64 v19; // [rsp+10h] [rbp-140h]
  _BYTE v20[312]; // [rsp+18h] [rbp-138h] BYREF

  v7 = a2;
  v8 = *((_BYTE *)a2 + 33) == 1;
  v17 = (const char **)v20;
  v18 = 0;
  v19 = 256;
  if ( !v8 )
    goto LABEL_6;
  v9 = *((unsigned __int8 *)a2 + 32);
  if ( (_BYTE)v9 == 1 )
  {
    v11 = 0;
    v10 = 0;
    goto LABEL_7;
  }
  a3 = (unsigned int)(v9 - 3);
  if ( (unsigned __int8)(v9 - 3) > 3u )
  {
LABEL_6:
    a2 = (const char **)&v17;
    sub_CA0EC0(v7, &v17);
    v11 = v18;
    v10 = v17;
    goto LABEL_7;
  }
  if ( (_BYTE)v9 == 4 )
  {
    v10 = *(const char ***)*a2;
    v11 = *((_QWORD *)*a2 + 1);
    goto LABEL_7;
  }
  if ( (unsigned __int8)v9 > 4u )
  {
    if ( (unsigned __int8)(v9 - 5) <= 1u )
    {
      v11 = (size_t)a2[1];
      v10 = (const char **)*a2;
      goto LABEL_7;
    }
LABEL_21:
    BUG();
  }
  if ( (_BYTE)v9 != 3 )
    goto LABEL_21;
  v10 = (const char **)*a2;
  v11 = 0;
  if ( *a2 )
    v11 = strlen(*a2);
LABEL_7:
  v12 = malloc(v11 + a1 + 9, a2, a3, a4, a5, a6);
  v13 = v12;
  if ( !v12 )
    sub_C64F00("Allocation failed", 1u);
  *(_QWORD *)(v12 + a1) = v11;
  v14 = (_BYTE *)(v12 + a1 + 8);
  if ( v11 )
  {
    a2 = v10;
    v14 = memcpy((void *)(v12 + a1 + 8), v10, v11);
  }
  v15 = v17;
  v14[v11] = 0;
  if ( v15 != (const char **)v20 )
    _libc_free(v15, a2);
  return v13;
}
