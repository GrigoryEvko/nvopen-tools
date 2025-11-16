// Function: sub_A53C60
// Address: 0xa53c60
//
void __fastcall sub_A53C60(__int64 *a1, const void *a2, size_t a3, unsigned int a4)
{
  __int64 v7; // r12
  void *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned int v10; // eax
  unsigned int *v11; // rdi
  unsigned int *v12; // r13
  unsigned int *v13; // r14
  char v14; // r12
  void *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r15
  const void *v18; // r10
  size_t v19; // rdx
  size_t v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  size_t v25; // [rsp-80h] [rbp-80h]
  unsigned int v26; // [rsp-74h] [rbp-74h]
  size_t v27; // [rsp-70h] [rbp-70h]
  const void *v28; // [rsp-70h] [rbp-70h]
  unsigned int *v29; // [rsp-68h] [rbp-68h] BYREF
  __int64 v30; // [rsp-60h] [rbp-60h]
  _BYTE v31[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( a4 )
  {
    v7 = *a1;
    if ( *((_BYTE *)a1 + 8) )
      *((_BYTE *)a1 + 8) = 0;
    else
      v7 = sub_904010(*a1, (const char *)a1[2]);
    v8 = *(void **)(v7 + 32);
    if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 < a3 )
    {
      v7 = sub_CB6200(v7, a2, a3);
    }
    else if ( a3 )
    {
      memcpy(v8, a2, a3);
      *(_QWORD *)(v7 + 32) += a3;
    }
    sub_904010(v7, ": ");
    v9 = (unsigned __int64)&v29;
    v29 = (unsigned int *)v31;
    v30 = 0x800000000LL;
    v10 = sub_AF5BC0(a4, &v29);
    v11 = v29;
    v26 = v10;
    v12 = &v29[(unsigned int)v30];
    if ( v29 != v12 )
    {
      v13 = v29;
      v14 = 1;
      while ( 1 )
      {
        v16 = sub_AF2320(*v13);
        v17 = *a1;
        v18 = (const void *)v16;
        v20 = v19;
        if ( v14 )
          break;
        v21 = *(_QWORD *)(v17 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v21) <= 2 )
        {
          v25 = v20;
          v28 = (const void *)v16;
          v23 = sub_CB6200(*a1, " | ", 3);
          v18 = v28;
          v20 = v25;
          v15 = *(void **)(v23 + 32);
          v17 = v23;
LABEL_10:
          if ( *(_QWORD *)(v17 + 24) - (_QWORD)v15 >= v20 )
            goto LABEL_11;
LABEL_17:
          ++v13;
          sub_CB6200(v17, v18, v20);
          if ( v12 == v13 )
            goto LABEL_18;
        }
        else
        {
          *(_BYTE *)(v21 + 2) = 32;
          *(_WORD *)v21 = 31776;
          v15 = (void *)(*(_QWORD *)(v17 + 32) + 3LL);
          v22 = *(_QWORD *)(v17 + 24);
          *(_QWORD *)(v17 + 32) = v15;
          if ( v22 - (__int64)v15 < v20 )
            goto LABEL_17;
LABEL_11:
          if ( v20 )
          {
            v27 = v20;
            memcpy(v15, v18, v20);
            *(_QWORD *)(v17 + 32) += v27;
          }
          if ( v12 == ++v13 )
          {
LABEL_18:
            v9 = v26;
            if ( !v26 && (_DWORD)v30 )
            {
LABEL_20:
              v11 = v29;
              goto LABEL_21;
            }
            v24 = sub_904010(*a1, " | ");
LABEL_27:
            v9 = v26;
            sub_CB59D0(v24, v26);
            goto LABEL_20;
          }
        }
      }
      v15 = *(void **)(v17 + 32);
      v14 = 0;
      goto LABEL_10;
    }
    if ( v10 || !(_DWORD)v30 )
    {
      v24 = *a1;
      goto LABEL_27;
    }
LABEL_21:
    if ( v11 != (unsigned int *)v31 )
      _libc_free(v11, v9);
  }
}
