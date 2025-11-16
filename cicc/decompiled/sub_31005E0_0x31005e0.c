// Function: sub_31005E0
// Address: 0x31005e0
//
void __fastcall sub_31005E0(
        __int64 a1,
        unsigned __int8 *a2,
        size_t a3,
        unsigned __int8 *a4,
        size_t a5,
        unsigned int a6)
{
  __int64 v8; // r12
  __int64 v10; // rdx
  void *v11; // rdi
  _BYTE *v12; // r15
  __int64 v13; // rdi
  _BYTE *v14; // r8
  _BYTE *v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // [rsp-40h] [rbp-40h]

  if ( !a3 )
    return;
  v8 = a1;
  v10 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v10) <= 5 )
  {
    v17 = a4;
    sub_CB6200(a1, (unsigned __int8 *)" from ", 6u);
    v11 = *(void **)(a1 + 32);
    a4 = v17;
    if ( !a5 )
      goto LABEL_4;
  }
  else
  {
    *(_DWORD *)v10 = 1869768224;
    *(_WORD *)(v10 + 4) = 8301;
    v11 = (void *)(*(_QWORD *)(a1 + 32) + 6LL);
    *(_QWORD *)(v8 + 32) = v11;
    if ( !a5 )
      goto LABEL_4;
  }
  if ( a5 > *(_QWORD *)(v8 + 24) - (_QWORD)v11 )
  {
    v16 = sub_CB6200(v8, a4, a5);
    v14 = *(_BYTE **)(v16 + 32);
    v13 = v16;
  }
  else
  {
    memcpy(v11, a4, a5);
    v12 = (_BYTE *)(*(_QWORD *)(v8 + 32) + a5);
    v13 = v8;
    *(_QWORD *)(v8 + 32) = v12;
    v14 = v12;
  }
  if ( v14 == *(_BYTE **)(v13 + 24) )
  {
    sub_CB6200(v13, (unsigned __int8 *)"/", 1u);
  }
  else
  {
    *v14 = 47;
    ++*(_QWORD *)(v13 + 32);
  }
  v11 = *(void **)(v8 + 32);
LABEL_4:
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v11 < a3 )
  {
    sub_CB6200(v8, a2, a3);
  }
  else
  {
    memcpy(v11, a2, a3);
    *(_QWORD *)(v8 + 32) += a3;
  }
  if ( a6 )
  {
    v15 = *(_BYTE **)(v8 + 32);
    if ( *(_BYTE **)(v8 + 24) == v15 )
    {
      v8 = sub_CB6200(v8, (unsigned __int8 *)":", 1u);
    }
    else
    {
      *v15 = 58;
      ++*(_QWORD *)(v8 + 32);
    }
    sub_CB59D0(v8, a6);
  }
}
