// Function: sub_A5CC00
// Address: 0xa5cc00
//
void __fastcall sub_A5CC00(__int64 a1, const void *a2, size_t a3, __int64 a4, char a5)
{
  __int64 v8; // r13
  _WORD *v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax

  if ( a4 || !a5 )
  {
    v8 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 8) )
    {
      *(_BYTE *)(a1 + 8) = 0;
      v9 = *(_WORD **)(v8 + 32);
      v10 = *(_QWORD *)(v8 + 24) - (_QWORD)v9;
      if ( v10 >= a3 )
        goto LABEL_5;
    }
    else
    {
      v8 = sub_904010(*(_QWORD *)a1, *(const char **)(a1 + 16));
      v9 = *(_WORD **)(v8 + 32);
      v10 = *(_QWORD *)(v8 + 24) - (_QWORD)v9;
      if ( v10 >= a3 )
      {
LABEL_5:
        if ( a3 )
        {
          memcpy(v9, a2, a3);
          v14 = *(_QWORD *)(v8 + 24);
          v9 = (_WORD *)(a3 + *(_QWORD *)(v8 + 32));
          *(_QWORD *)(v8 + 32) = v9;
          v10 = v14 - (_QWORD)v9;
        }
        if ( v10 > 1 )
        {
LABEL_8:
          *v9 = 8250;
          *(_QWORD *)(v8 + 32) += 2LL;
          v11 = *(_QWORD *)a1;
          if ( a4 )
          {
LABEL_9:
            v12 = *(__int64 **)(a1 + 24);
            sub_A5C090(v11, a4, v12);
            (*(void (__fastcall **)(__int64 *, __int64))*v12)(v12, a4);
            return;
          }
          goto LABEL_13;
        }
LABEL_12:
        sub_CB6200(v8, ": ", 2);
        v11 = *(_QWORD *)a1;
        if ( a4 )
          goto LABEL_9;
LABEL_13:
        sub_904010(v11, "null");
        return;
      }
    }
    v13 = sub_CB6200(v8, a2, a3);
    v9 = *(_WORD **)(v13 + 32);
    v8 = v13;
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v9 > 1u )
      goto LABEL_8;
    goto LABEL_12;
  }
}
