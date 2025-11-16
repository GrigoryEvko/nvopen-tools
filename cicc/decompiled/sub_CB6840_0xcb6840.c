// Function: sub_CB6840
// Address: 0xcb6840
//
__int64 __fastcall sub_CB6840(__int64 a1, __int64 a2)
{
  char *v4; // rsi
  char *v5; // rbx
  char *v6; // r13
  __int64 v7; // rsi
  int v8; // edx
  char v9; // al
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  void *v13; // rdi
  size_t v14; // rdx
  size_t v16; // [rsp+8h] [rbp-D8h]
  __int64 v17; // [rsp+10h] [rbp-D0h] BYREF
  int v18; // [rsp+18h] [rbp-C8h]
  int v19; // [rsp+1Ch] [rbp-C4h]
  char v20; // [rsp+20h] [rbp-C0h]
  char *v21; // [rsp+30h] [rbp-B0h] BYREF
  int v22; // [rsp+38h] [rbp-A8h]
  char v23; // [rsp+40h] [rbp-A0h] BYREF

  v4 = *(char **)a2;
  sub_C65DD0((__int64)&v21, v4, *(_QWORD *)(a2 + 8));
  v5 = v21;
  v6 = &v21[56 * v22];
  if ( v21 != v6 )
  {
    while ( 1 )
    {
      while ( *(_DWORD *)v5 == 1 )
      {
        v13 = *(void **)(a1 + 32);
        v14 = *((_QWORD *)v5 + 2);
        v4 = (char *)*((_QWORD *)v5 + 1);
        if ( *(_QWORD *)(a1 + 24) - (_QWORD)v13 < v14 )
        {
LABEL_15:
          sub_CB6200(a1, (unsigned __int8 *)v4, v14);
          goto LABEL_4;
        }
LABEL_8:
        if ( !v14 )
          goto LABEL_4;
        v5 += 56;
        v16 = v14;
        memcpy(v13, v4, v14);
        *(_QWORD *)(a1 + 32) += v16;
        if ( v6 == v5 )
        {
LABEL_10:
          v6 = v21;
          goto LABEL_11;
        }
      }
      v12 = *((unsigned int *)v5 + 6);
      if ( v12 >= *(_QWORD *)(a2 + 24) )
      {
        v13 = *(void **)(a1 + 32);
        v14 = *((_QWORD *)v5 + 2);
        v4 = (char *)*((_QWORD *)v5 + 1);
        if ( v14 > *(_QWORD *)(a1 + 24) - (_QWORD)v13 )
          goto LABEL_15;
        goto LABEL_8;
      }
      v7 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v12);
      v8 = *((_DWORD *)v5 + 7);
      v9 = v5[36];
      v18 = *((_DWORD *)v5 + 8);
      v19 = v8;
      v10 = *((_QWORD *)v5 + 6);
      v11 = *((_QWORD *)v5 + 5);
      v17 = v7;
      v4 = (char *)a1;
      v20 = v9;
      sub_CB6320(&v17, a1, v11, v10);
LABEL_4:
      v5 += 56;
      if ( v6 == v5 )
        goto LABEL_10;
    }
  }
LABEL_11:
  if ( v6 != &v23 )
    _libc_free(v6, v4);
  return a1;
}
