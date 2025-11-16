// Function: sub_A53AC0
// Address: 0xa53ac0
//
void __fastcall sub_A53AC0(
        __int64 *a1,
        const void *a2,
        size_t a3,
        unsigned int a4,
        __int64 (__fastcall *a5)(_QWORD),
        char a6)
{
  const void *v6; // r10
  __int64 v11; // r13
  _WORD *v12; // rdi
  unsigned __int64 v13; // rax
  const void *v14; // rax
  size_t v15; // rdx
  size_t v16; // r13
  __int64 v17; // r12
  void *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  size_t v21; // rdx
  __int64 v22; // rax

  v6 = a2;
  if ( a4 || !a6 )
  {
    v11 = *a1;
    if ( *((_BYTE *)a1 + 8) )
    {
      *((_BYTE *)a1 + 8) = 0;
      v12 = *(_WORD **)(v11 + 32);
      v13 = *(_QWORD *)(v11 + 24) - (_QWORD)v12;
      if ( v13 >= a3 )
        goto LABEL_5;
    }
    else
    {
      v19 = sub_904010(*a1, (const char *)a1[2]);
      v6 = a2;
      v11 = v19;
      v12 = *(_WORD **)(v19 + 32);
      v13 = *(_QWORD *)(v19 + 24) - (_QWORD)v12;
      if ( v13 >= a3 )
      {
LABEL_5:
        if ( a3 )
        {
          memcpy(v12, v6, a3);
          v22 = *(_QWORD *)(v11 + 24);
          v12 = (_WORD *)(a3 + *(_QWORD *)(v11 + 32));
          *(_QWORD *)(v11 + 32) = v12;
          v13 = v22 - (_QWORD)v12;
        }
        if ( v13 > 1 )
        {
LABEL_8:
          *v12 = 8250;
          *(_QWORD *)(v11 + 32) += 2LL;
          v14 = (const void *)a5(a4);
          v16 = v15;
          if ( v15 )
            goto LABEL_9;
          goto LABEL_14;
        }
LABEL_13:
        sub_CB6200(v11, ": ", 2);
        v14 = (const void *)a5(a4);
        v16 = v21;
        if ( v21 )
        {
LABEL_9:
          v17 = *a1;
          v18 = *(void **)(v17 + 32);
          if ( v16 > *(_QWORD *)(v17 + 24) - (_QWORD)v18 )
          {
            sub_CB6200(v17, v14, v16);
          }
          else
          {
            memcpy(v18, v14, v16);
            *(_QWORD *)(v17 + 32) += v16;
          }
          return;
        }
LABEL_14:
        sub_CB59D0(*a1, a4);
        return;
      }
    }
    v20 = sub_CB6200(v11, v6, a3);
    v12 = *(_WORD **)(v20 + 32);
    v11 = v20;
    if ( *(_QWORD *)(v20 + 24) - (_QWORD)v12 > 1u )
      goto LABEL_8;
    goto LABEL_13;
  }
}
