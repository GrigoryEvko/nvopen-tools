// Function: sub_14231B0
// Address: 0x14231b0
//
_BYTE *__fastcall sub_14231B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  void *v5; // rdx
  char v6; // al
  __int64 v7; // r14
  __int64 *v8; // rbx
  __int64 *v9; // r14
  char i; // cl
  _BYTE *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r9
  _BYTE *v15; // rax
  __int64 v16; // rax
  size_t v17; // rdx
  _BYTE *v18; // rdi
  const char *v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  void *v22; // rdx
  _BYTE *result; // rax
  _BYTE *v24; // rax
  __int64 v25; // [rsp+8h] [rbp-38h]
  size_t v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v4 = sub_16E7A90(a2, *(unsigned int *)(a1 + 72));
  v5 = *(void **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0xCu )
  {
    sub_16E7EE0(v4, " = MemoryPhi(", 13);
  }
  else
  {
    qmemcpy(v5, " = MemoryPhi(", 13);
    *(_QWORD *)(v4 + 24) += 13LL;
  }
  v6 = *(_BYTE *)(a1 + 23);
  v7 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (v6 & 0x40) != 0 )
  {
    v8 = *(__int64 **)(a1 - 8);
    v9 = &v8[v7];
  }
  else
  {
    v8 = (__int64 *)(a1 - v7 * 8);
    v9 = (__int64 *)a1;
  }
  if ( v8 != v9 )
  {
    for ( i = 1; ; i = 0 )
    {
      if ( (v6 & 0x40) != 0 )
        v12 = *(_QWORD *)(a1 - 8);
      else
        v12 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v13 = *v8;
      v14 = *(_QWORD *)(v12
                      + 0xFFFFFFFD55555558LL * (unsigned int)(((__int64)v8 - v12) >> 3)
                      + 24LL * *(unsigned int *)(a1 + 76)
                      + 8);
      if ( !i )
      {
        v24 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v24 >= *(_QWORD *)(a2 + 16) )
        {
          v27 = *(_QWORD *)(v12
                          + 0xFFFFFFFD55555558LL * (unsigned int)(((__int64)v8 - v12) >> 3)
                          + 24LL * *(unsigned int *)(a1 + 76)
                          + 8);
          sub_16E7DE0(a2, 44);
          v14 = v27;
        }
        else
        {
          *(_QWORD *)(a2 + 24) = v24 + 1;
          *v24 = 44;
        }
      }
      v15 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(a2 + 16) )
      {
        v25 = v14;
        sub_16E7DE0(a2, 123);
        v14 = v25;
      }
      else
      {
        *(_QWORD *)(a2 + 24) = v15 + 1;
        *v15 = 123;
      }
      if ( (*(_BYTE *)(v14 + 23) & 0x20) != 0 )
      {
        v16 = sub_1649960(v14);
        v18 = *(_BYTE **)(a2 + 24);
        v19 = (const char *)v16;
        v20 = *(_QWORD *)(a2 + 16);
        if ( v17 > v20 - (unsigned __int64)v18 )
        {
          sub_16E7EE0(a2, v19);
          v18 = *(_BYTE **)(a2 + 24);
          v20 = *(_QWORD *)(a2 + 16);
        }
        else if ( v17 )
        {
          v26 = v17;
          memcpy(v18, v19, v17);
          v20 = *(_QWORD *)(a2 + 16);
          v18 = (_BYTE *)(v26 + *(_QWORD *)(a2 + 24));
          *(_QWORD *)(a2 + 24) = v18;
        }
        if ( (unsigned __int64)v18 < v20 )
        {
LABEL_21:
          *(_QWORD *)(a2 + 24) = v18 + 1;
          *v18 = 44;
          if ( *(_BYTE *)(v13 + 16) == 22 )
            goto LABEL_22;
          goto LABEL_31;
        }
      }
      else
      {
        sub_15537D0(v14, a2, 0);
        v18 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v18 < *(_QWORD *)(a2 + 16) )
          goto LABEL_21;
      }
      sub_16E7DE0(a2, 44);
      if ( *(_BYTE *)(v13 + 16) == 22 )
      {
LABEL_22:
        v21 = *(unsigned int *)(v13 + 84);
        goto LABEL_23;
      }
LABEL_31:
      v21 = *(unsigned int *)(v13 + 72);
LABEL_23:
      if ( (_DWORD)v21 )
      {
        sub_16E7A90(a2, v21);
        v11 = *(_BYTE **)(a2 + 24);
        goto LABEL_8;
      }
      v22 = *(void **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v22 <= 0xAu )
      {
        sub_16E7EE0(a2, "liveOnEntry", 11);
        v11 = *(_BYTE **)(a2 + 24);
LABEL_8:
        if ( (unsigned __int64)v11 < *(_QWORD *)(a2 + 16) )
          goto LABEL_9;
        goto LABEL_26;
      }
      qmemcpy(v22, "liveOnEntry", 11);
      v11 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 11LL);
      *(_QWORD *)(a2 + 24) = v11;
      if ( (unsigned __int64)v11 < *(_QWORD *)(a2 + 16) )
      {
LABEL_9:
        v8 += 3;
        *(_QWORD *)(a2 + 24) = v11 + 1;
        *v11 = 125;
        if ( v8 == v9 )
          break;
        goto LABEL_10;
      }
LABEL_26:
      v8 += 3;
      sub_16E7DE0(a2, 125);
      if ( v8 == v9 )
        break;
LABEL_10:
      v6 = *(_BYTE *)(a1 + 23);
    }
  }
  result = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 16) )
    return (_BYTE *)sub_16E7DE0(a2, 41);
  *(_QWORD *)(a2 + 24) = result + 1;
  *result = 41;
  return result;
}
