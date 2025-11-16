// Function: sub_E401D0
// Address: 0xe401d0
//
unsigned __int64 __fastcall sub_E401D0(__int64 a1, char *a2, int a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v8; // rdi
  bool v10; // zf
  unsigned __int64 result; // rax
  char *v12; // r10
  size_t v13; // r9
  char v14; // dl
  int v15; // eax
  unsigned __int8 *v16; // rdi
  size_t v17; // rdx
  void *v18; // rdi
  size_t v19; // r13
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rcx
  char *v23; // [rsp+0h] [rbp-160h]
  char *v24; // [rsp+0h] [rbp-160h]
  size_t v25; // [rsp+8h] [rbp-158h]
  size_t v26; // [rsp+8h] [rbp-158h]
  size_t v27; // [rsp+8h] [rbp-158h]
  char *v28; // [rsp+8h] [rbp-158h]
  char *v29; // [rsp+10h] [rbp-150h] BYREF
  size_t v30; // [rsp+18h] [rbp-148h]
  __int64 v31; // [rsp+20h] [rbp-140h]
  _BYTE v32[312]; // [rsp+28h] [rbp-138h] BYREF

  v8 = (__int64)a2;
  v10 = a2[33] == 1;
  v29 = v32;
  v30 = 0;
  v31 = 256;
  if ( !v10 )
    goto LABEL_6;
  result = (unsigned __int8)a2[32];
  if ( (_BYTE)result == 1 )
  {
    v14 = MEMORY[0];
    if ( MEMORY[0] == 1 )
      goto LABEL_17;
    v12 = 0;
    v13 = 0;
    goto LABEL_8;
  }
  if ( (unsigned __int8)(result - 3) > 3u )
  {
LABEL_6:
    a2 = (char *)&v29;
    result = sub_CA0EC0(v8, (__int64)&v29);
    v13 = v30;
    v12 = v29;
    goto LABEL_7;
  }
  if ( (_BYTE)result == 4 )
  {
    result = *(_QWORD *)a2;
    v12 = **(char ***)a2;
    v13 = *(_QWORD *)(*(_QWORD *)a2 + 8LL);
    goto LABEL_7;
  }
  if ( (unsigned __int8)result > 4u )
  {
    result = (unsigned int)(result - 5);
    if ( (unsigned __int8)result <= 1u )
    {
      v13 = *((_QWORD *)a2 + 1);
      v12 = *(char **)a2;
      goto LABEL_7;
    }
LABEL_56:
    BUG();
  }
  if ( (_BYTE)result != 3 )
    goto LABEL_56;
  v12 = *(char **)a2;
  if ( !*(_QWORD *)a2 )
  {
    v14 = MEMORY[0];
    if ( MEMORY[0] == 1 )
      goto LABEL_17;
    v13 = 0;
LABEL_8:
    v15 = *(_DWORD *)(a4 + 24);
    if ( (unsigned int)(v15 - 3) <= 1 && v14 == 63 )
      a5 = 0;
    if ( a3 == 1 )
    {
      switch ( v15 )
      {
        case 0:
          v16 = *(unsigned __int8 **)(a1 + 32);
          break;
        case 1:
        case 3:
          v17 = 2;
          a2 = ".L";
          goto LABEL_43;
        case 2:
        case 4:
          v17 = 1;
          a2 = "L";
          goto LABEL_43;
        case 5:
          v17 = 2;
          a2 = "L#";
          goto LABEL_43;
        case 6:
          v17 = 1;
          a2 = "$";
          goto LABEL_43;
        case 7:
          v17 = 3;
          a2 = "L..";
LABEL_43:
          v20 = *(_QWORD *)(a1 + 32);
          if ( *(_QWORD *)(a1 + 24) - v20 < v17 )
          {
            v23 = v12;
            v25 = v13;
            goto LABEL_23;
          }
          v21 = 0;
          do
          {
            v22 = v21++;
            *(_BYTE *)(v20 + v22) = a2[v22];
          }
          while ( v21 < (unsigned int)v17 );
          v16 = (unsigned __int8 *)(v17 + *(_QWORD *)(a1 + 32));
          *(_QWORD *)(a1 + 32) = v16;
          break;
        default:
          goto LABEL_56;
      }
    }
    else
    {
      v16 = *(unsigned __int8 **)(a1 + 32);
      if ( a3 == 2 && v15 == 2 )
      {
        if ( v16 == *(unsigned __int8 **)(a1 + 24) )
        {
          v23 = v12;
          v17 = 1;
          a2 = "l";
          v25 = v13;
LABEL_23:
          sub_CB6200(a1, (unsigned __int8 *)a2, v17);
          v16 = *(unsigned __int8 **)(a1 + 32);
          v13 = v25;
          v12 = v23;
          if ( !a5 )
            goto LABEL_14;
          goto LABEL_24;
        }
        *v16 = 108;
        v16 = (unsigned __int8 *)(*(_QWORD *)(a1 + 32) + 1LL);
        *(_QWORD *)(a1 + 32) = v16;
      }
    }
    if ( !a5 )
      goto LABEL_14;
LABEL_24:
    if ( (unsigned __int64)v16 < *(_QWORD *)(a1 + 24) )
    {
      *(_QWORD *)(a1 + 32) = v16 + 1;
      *v16 = a5;
      v16 = *(unsigned __int8 **)(a1 + 32);
      result = *(_QWORD *)(a1 + 24) - (_QWORD)v16;
      if ( result < v13 )
        goto LABEL_26;
LABEL_15:
      if ( v13 )
      {
        a2 = v12;
        v26 = v13;
        result = (unsigned __int64)memcpy(v16, v12, v13);
        *(_QWORD *)(a1 + 32) += v26;
      }
      goto LABEL_17;
    }
    a2 = (char *)a5;
    v24 = v12;
    v27 = v13;
    sub_CB5D20(a1, a5);
    v16 = *(unsigned __int8 **)(a1 + 32);
    v13 = v27;
    v12 = v24;
LABEL_14:
    result = *(_QWORD *)(a1 + 24) - (_QWORD)v16;
    if ( result < v13 )
    {
LABEL_26:
      a2 = v12;
      result = sub_CB6200(a1, (unsigned __int8 *)v12, v13);
      goto LABEL_17;
    }
    goto LABEL_15;
  }
  v28 = *(char **)a2;
  result = strlen(*(const char **)a2);
  v12 = v28;
  v13 = result;
LABEL_7:
  v14 = *v12;
  if ( *v12 != 1 )
    goto LABEL_8;
  if ( v13 )
  {
    v18 = *(void **)(a1 + 32);
    v19 = v13 - 1;
    a2 = v12 + 1;
    result = *(_QWORD *)(a1 + 24) - (_QWORD)v18;
    if ( result < v13 - 1 )
    {
      result = sub_CB6200(a1, (unsigned __int8 *)a2, v13 - 1);
    }
    else if ( v13 != 1 )
    {
      result = (unsigned __int64)memcpy(v18, a2, v13 - 1);
      *(_QWORD *)(a1 + 32) += v19;
    }
  }
LABEL_17:
  if ( v29 != v32 )
    return _libc_free(v29, a2);
  return result;
}
