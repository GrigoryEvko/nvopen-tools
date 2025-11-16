// Function: sub_E4E8B0
// Address: 0xe4e8b0
//
_BYTE *__fastcall sub_E4E8B0(__int64 a1, __int64 a2, char *a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdi
  _BYTE *v10; // rax
  _BYTE *v11; // rax
  char *v12; // r14
  char v13; // r12
  __int64 v14; // rdi
  char *v15; // rax
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v19; // rdi
  _BYTE *v20; // rax

  v6 = *(_QWORD *)(a1 + 304);
  v8 = *(_QWORD *)(v6 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 24) - v8) <= 8 )
  {
    sub_CB6200(v6, "\t.rename\t", 9u);
  }
  else
  {
    *(_BYTE *)(v8 + 8) = 9;
    *(_QWORD *)v8 = 0x656D616E65722E09LL;
    *(_QWORD *)(v6 + 32) += 9LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(_BYTE **)(v9 + 32);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
  {
    v9 = sub_CB5D20(v9, 44);
  }
  else
  {
    *(_QWORD *)(v9 + 32) = v10 + 1;
    *v10 = 44;
  }
  v11 = *(_BYTE **)(v9 + 32);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(v9 + 24) )
  {
    sub_CB5D20(v9, 34);
  }
  else
  {
    *(_QWORD *)(v9 + 32) = v11 + 1;
    *v11 = 34;
  }
  v12 = &a3[a4];
  while ( v12 != a3 )
  {
    while ( 1 )
    {
      v13 = *a3;
      if ( *a3 == 34 )
      {
        v19 = *(_QWORD *)(a1 + 304);
        v20 = *(_BYTE **)(v19 + 32);
        if ( *(_QWORD *)(v19 + 24) <= (unsigned __int64)v20 )
        {
          sub_CB5D20(v19, 34);
        }
        else
        {
          *(_QWORD *)(v19 + 32) = v20 + 1;
          *v20 = 34;
        }
      }
      v14 = *(_QWORD *)(a1 + 304);
      v15 = *(char **)(v14 + 32);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 24) )
        break;
      ++a3;
      *(_QWORD *)(v14 + 32) = v15 + 1;
      *v15 = v13;
      if ( v12 == a3 )
        goto LABEL_13;
    }
    ++a3;
    sub_CB5D20(v14, v13);
  }
LABEL_13:
  v16 = *(_QWORD *)(a1 + 304);
  v17 = *(_BYTE **)(v16 + 32);
  if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 24) )
  {
    sub_CB5D20(v16, 34);
  }
  else
  {
    *(_QWORD *)(v16 + 32) = v17 + 1;
    *v17 = 34;
  }
  return sub_E4D880(a1);
}
