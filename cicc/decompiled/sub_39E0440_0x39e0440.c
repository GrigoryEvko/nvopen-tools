// Function: sub_39E0440
// Address: 0x39e0440
//
_BYTE *__fastcall sub_39E0440(__int64 a1)
{
  __int64 v2; // r12
  char *v3; // r14
  size_t v4; // rdx
  _BYTE *v5; // rax
  size_t v6; // r9
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  char *v10; // rsi
  size_t v11; // rdx
  unsigned __int64 v12; // rax
  _BYTE *v13; // rdi
  unsigned __int64 v14; // rax
  _BYTE *v15; // rdi
  _BYTE *result; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  _BYTE *v20; // rdx
  __int64 v21; // rdi
  size_t v22; // [rsp+0h] [rbp-40h]
  size_t v23; // [rsp+8h] [rbp-38h]
  size_t v24; // [rsp+8h] [rbp-38h]
  size_t v25; // [rsp+8h] [rbp-38h]
  size_t v26; // [rsp+8h] [rbp-38h]

  v2 = *(unsigned int *)(a1 + 456);
  if ( !(_DWORD)v2 && *(_QWORD *)(a1 + 616) == *(_QWORD *)(a1 + 600) )
  {
    v21 = *(_QWORD *)(a1 + 272);
    result = *(_BYTE **)(v21 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(v21 + 16) )
    {
      return (_BYTE *)sub_16E7DE0(v21, 10);
    }
    else
    {
      *(_QWORD *)(v21 + 24) = result + 1;
      *result = 10;
    }
  }
  else
  {
    v3 = *(char **)(a1 + 448);
    do
    {
      sub_16BE270(*(_QWORD *)(a1 + 272), 40);
      if ( v2 )
      {
        v4 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v2 >= 0 )
          v4 = v2;
        v5 = memchr(v3, 10, v4);
        if ( v5 )
        {
          v6 = v2;
          if ( v5 - v3 <= (unsigned __int64)v2 )
            v6 = v5 - v3;
          v7 = v5 - v3 + 1;
        }
        else
        {
          v6 = v2;
          v7 = 0;
        }
      }
      else
      {
        v7 = 0;
        v6 = 0;
      }
      v8 = *(_QWORD *)(a1 + 280);
      v9 = *(_QWORD *)(a1 + 272);
      v10 = *(char **)(v8 + 48);
      v11 = *(_QWORD *)(v8 + 56);
      v12 = *(_QWORD *)(v9 + 16);
      v13 = *(_BYTE **)(v9 + 24);
      if ( v11 > v12 - (unsigned __int64)v13 )
      {
        v24 = v6;
        v19 = sub_16E7EE0(*(_QWORD *)(a1 + 272), v10, v11);
        v6 = v24;
        v13 = *(_BYTE **)(v19 + 24);
        v9 = v19;
        v12 = *(_QWORD *)(v19 + 16);
      }
      else if ( v11 )
      {
        v22 = v6;
        v25 = v11;
        memcpy(v13, v10, v11);
        v20 = (_BYTE *)(*(_QWORD *)(v9 + 24) + v25);
        v12 = *(_QWORD *)(v9 + 16);
        v6 = v22;
        *(_QWORD *)(v9 + 24) = v20;
        v13 = v20;
      }
      if ( (unsigned __int64)v13 >= v12 )
      {
        v23 = v6;
        v18 = sub_16E7DE0(v9, 32);
        v6 = v23;
        v9 = v18;
      }
      else
      {
        *(_QWORD *)(v9 + 24) = v13 + 1;
        *v13 = 32;
      }
      v14 = *(_QWORD *)(v9 + 16);
      v15 = *(_BYTE **)(v9 + 24);
      if ( v14 - (unsigned __int64)v15 < v6 )
      {
        v17 = sub_16E7EE0(v9, v3, v6);
        v15 = *(_BYTE **)(v17 + 24);
        v9 = v17;
        v14 = *(_QWORD *)(v17 + 16);
      }
      else if ( v6 )
      {
        v26 = v6;
        memcpy(v15, v3, v6);
        v14 = *(_QWORD *)(v9 + 16);
        v15 = (_BYTE *)(v26 + *(_QWORD *)(v9 + 24));
        *(_QWORD *)(v9 + 24) = v15;
      }
      if ( v14 <= (unsigned __int64)v15 )
      {
        result = (_BYTE *)sub_16E7DE0(v9, 10);
      }
      else
      {
        result = v15 + 1;
        *(_QWORD *)(v9 + 24) = v15 + 1;
        *v15 = 10;
      }
      if ( v7 > v2 )
        break;
      v2 -= v7;
      v3 += v7;
    }
    while ( v2 );
    *(_DWORD *)(a1 + 456) = 0;
  }
  return result;
}
