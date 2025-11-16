// Function: sub_30F52A0
// Address: 0x30f52a0
//
__int64 __fastcall sub_30F52A0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // r15
  __int64 v6; // rdi
  void *v7; // rdi
  char *v8; // rsi
  size_t v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  const char *v17; // rax
  __int64 v18; // rax
  void *v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-38h]
  size_t v21; // [rsp+8h] [rbp-38h]

  v3 = *(__int64 **)(a2 + 144);
  v4 = &v3[3 * *(unsigned int *)(a2 + 152)];
  if ( v4 != v3 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(a1 + 32);
      v14 = *v3;
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v13) > 5 )
      {
        *(_DWORD *)v13 = 1886351180;
        v5 = a1;
        *(_WORD *)(v13 + 4) = 10016;
        *(_QWORD *)(a1 + 32) += 6LL;
      }
      else
      {
        v20 = *v3;
        v15 = sub_CB6200(a1, (unsigned __int8 *)"Loop '", 6u);
        v14 = v20;
        v5 = v15;
      }
      v6 = **(_QWORD **)(v14 + 32);
      if ( !v6 || (*(_BYTE *)(v6 + 7) & 0x10) == 0 )
        break;
      v17 = sub_BD5D20(v6);
      v7 = *(void **)(v5 + 32);
      v8 = (char *)v17;
      v11 = *(_QWORD *)(v5 + 24) - (_QWORD)v7;
      if ( v11 < v9 )
        goto LABEL_7;
      if ( v9 )
      {
LABEL_19:
        v21 = v9;
        memcpy(v7, v8, v9);
        v18 = *(_QWORD *)(v5 + 24);
        v19 = (void *)(*(_QWORD *)(v5 + 32) + v21);
        *(_QWORD *)(v5 + 32) = v19;
        v7 = v19;
        v11 = v18 - (_QWORD)v19;
      }
LABEL_8:
      if ( v11 <= 0xC )
      {
        v5 = sub_CB6200(v5, "' has cost = ", 0xDu);
      }
      else
      {
        qmemcpy(v7, "' has cost = ", 13);
        *(_QWORD *)(v5 + 32) += 13LL;
      }
      sub_C68B50((__int64)(v3 + 1), v5);
      v12 = *(_BYTE **)(v5 + 32);
      if ( *(_BYTE **)(v5 + 24) == v12 )
      {
        v3 += 3;
        sub_CB6200(v5, (unsigned __int8 *)"\n", 1u);
        if ( v4 == v3 )
          return a1;
      }
      else
      {
        v3 += 3;
        *v12 = 10;
        ++*(_QWORD *)(v5 + 32);
        if ( v4 == v3 )
          return a1;
      }
    }
    v7 = *(void **)(v5 + 32);
    v8 = "<unnamed loop>";
    v9 = 14;
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v7 > 0xDu )
      goto LABEL_19;
LABEL_7:
    v10 = sub_CB6200(v5, (unsigned __int8 *)v8, v9);
    v7 = *(void **)(v10 + 32);
    v5 = v10;
    v11 = *(_QWORD *)(v10 + 24) - (_QWORD)v7;
    goto LABEL_8;
  }
  return a1;
}
