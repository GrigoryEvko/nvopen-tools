// Function: sub_A538D0
// Address: 0xa538d0
//
__int64 __fastcall sub_A538D0(__int64 a1, const void *a2, size_t a3, __int64 a4)
{
  __int64 v6; // r12
  _WORD *v7; // rdi
  unsigned __int64 v8; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  v6 = *(_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 8) )
    *(_BYTE *)(a1 + 8) = 0;
  else
    v6 = sub_904010(*(_QWORD *)a1, *(const char **)(a1 + 16));
  v7 = *(_WORD **)(v6 + 32);
  v8 = *(_QWORD *)(v6 + 24) - (_QWORD)v7;
  if ( v8 >= a3 )
  {
    if ( a3 )
    {
      memcpy(v7, a2, a3);
      v11 = *(_QWORD *)(v6 + 24);
      v7 = (_WORD *)(a3 + *(_QWORD *)(v6 + 32));
      *(_QWORD *)(v6 + 32) = v7;
      v8 = v11 - (_QWORD)v7;
    }
    if ( v8 > 1 )
      goto LABEL_7;
LABEL_10:
    v6 = sub_CB6200(v6, ": ", 2);
    return sub_CB59F0(v6, a4);
  }
  v10 = sub_CB6200(v6, a2, a3);
  v7 = *(_WORD **)(v10 + 32);
  v6 = v10;
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v7 <= 1u )
    goto LABEL_10;
LABEL_7:
  *v7 = 8250;
  *(_QWORD *)(v6 + 32) += 2LL;
  return sub_CB59F0(v6, a4);
}
