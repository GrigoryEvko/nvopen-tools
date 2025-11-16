// Function: sub_C088A0
// Address: 0xc088a0
//
__int64 __fastcall sub_C088A0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v5; // r13
  void *v6; // rax
  const char *v7; // rax
  size_t v8; // rdx
  void *v9; // rdi
  size_t v10; // r12
  _BYTE *v11; // rax

  v3 = *(_QWORD **)(a1 + 176);
  if ( !(unsigned __int8)sub_C05FA0(v3, a2) && *(_BYTE *)(a1 + 184) )
  {
    v5 = sub_CB72A0(v3, a2);
    v6 = *(void **)(v5 + 32);
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xBu )
    {
      v5 = sub_CB6200(v5, "in function ", 12);
    }
    else
    {
      qmemcpy(v6, "in function ", 12);
      *(_QWORD *)(v5 + 32) += 12LL;
    }
    v7 = sub_BD5D20(a2);
    v9 = *(void **)(v5 + 32);
    v10 = v8;
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v9 < v8 )
    {
      v5 = sub_CB6200(v5, v7, v8);
    }
    else if ( v8 )
    {
      memcpy(v9, v7, v8);
      *(_QWORD *)(v5 + 32) += v10;
    }
    v11 = *(_BYTE **)(v5 + 32);
    if ( (unsigned __int64)v11 >= *(_QWORD *)(v5 + 24) )
    {
      sub_CB5D20(v5, 10);
    }
    else
    {
      *(_QWORD *)(v5 + 32) = v11 + 1;
      *v11 = 10;
    }
    sub_C64ED0("Broken function found, compilation aborted!", 1);
  }
  return 0;
}
