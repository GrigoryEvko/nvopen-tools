// Function: sub_154B110
// Address: 0x154b110
//
void __fastcall sub_154B110(__int64 *a1, const char *a2, size_t a3, __int64 a4, __int64 (__fastcall *a5)(_QWORD))
{
  const char *v5; // r9
  unsigned int v9; // ebx
  __int64 v10; // r13
  _WORD *v11; // rdi
  unsigned __int64 v12; // rax
  const char *v13; // rax
  size_t v14; // rdx
  size_t v15; // r13
  __int64 v16; // r12
  void *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  size_t v20; // rdx
  __int64 v21; // rax
  const char *v22; // [rsp-40h] [rbp-40h]

  if ( (_DWORD)a4 )
  {
    v5 = a2;
    v9 = a4;
    v10 = *a1;
    if ( *((_BYTE *)a1 + 8) )
    {
      *((_BYTE *)a1 + 8) = 0;
      v11 = *(_WORD **)(v10 + 24);
      v12 = *(_QWORD *)(v10 + 16) - (_QWORD)v11;
      if ( v12 >= a3 )
        goto LABEL_4;
    }
    else
    {
      v22 = a2;
      a2 = (const char *)a1[2];
      v18 = sub_1263B40(v10, a2);
      v5 = v22;
      v10 = v18;
      v11 = *(_WORD **)(v18 + 24);
      v12 = *(_QWORD *)(v18 + 16) - (_QWORD)v11;
      if ( v12 >= a3 )
      {
LABEL_4:
        if ( a3 )
        {
          a2 = v5;
          memcpy(v11, v5, a3);
          v21 = *(_QWORD *)(v10 + 16);
          v11 = (_WORD *)(a3 + *(_QWORD *)(v10 + 24));
          *(_QWORD *)(v10 + 24) = v11;
          v12 = v21 - (_QWORD)v11;
        }
        if ( v12 > 1 )
        {
LABEL_7:
          *v11 = 8250;
          *(_QWORD *)(v10 + 24) += 2LL;
          v13 = (const char *)((__int64 (__fastcall *)(_QWORD, const char *, size_t, __int64, __int64 (__fastcall *)(_QWORD), const char *))a5)(
                                v9,
                                a2,
                                a3,
                                a4,
                                a5,
                                v5);
          v15 = v14;
          if ( v14 )
            goto LABEL_8;
          goto LABEL_13;
        }
LABEL_12:
        sub_16E7EE0(v10, ": ", 2, a4, a5, v5);
        v13 = (const char *)a5(v9);
        v15 = v20;
        if ( v20 )
        {
LABEL_8:
          v16 = *a1;
          v17 = *(void **)(v16 + 24);
          if ( v15 > *(_QWORD *)(v16 + 16) - (_QWORD)v17 )
          {
            sub_16E7EE0(v16, v13, v15);
          }
          else
          {
            memcpy(v17, v13, v15);
            *(_QWORD *)(v16 + 24) += v15;
          }
          return;
        }
LABEL_13:
        sub_16E7A90(*a1, v9);
        return;
      }
    }
    a2 = v5;
    v19 = sub_16E7EE0(v10, v5, a3);
    v11 = *(_WORD **)(v19 + 24);
    v10 = v19;
    if ( *(_QWORD *)(v19 + 16) - (_QWORD)v11 > 1u )
      goto LABEL_7;
    goto LABEL_12;
  }
}
