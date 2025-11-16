// Function: sub_38E2490
// Address: 0x38e2490
//
void __fastcall sub_38E2490(_BYTE *a1, __int64 a2, _BYTE *a3)
{
  char *v5; // r15
  size_t v6; // r14
  _QWORD *v7; // rbx
  _BYTE *v8; // rax
  char *v9; // r14
  _WORD *i; // rax
  char v11; // si
  unsigned __int64 v12; // rdx
  void *v13; // rdi

  if ( (*a1 & 4) == 0 )
  {
    if ( !a3 || (*(unsigned __int8 (__fastcall **)(_BYTE *, _QWORD, _QWORD))(*(_QWORD *)a3 + 48LL))(a3, 0, 0) )
      return;
    v5 = 0;
    v6 = 0;
    goto LABEL_10;
  }
  v7 = (_QWORD *)*((_QWORD *)a1 - 1);
  v6 = *v7;
  v5 = (char *)(v7 + 2);
  if ( a3 && !(*(unsigned __int8 (__fastcall **)(_BYTE *, _QWORD *, _QWORD))(*(_QWORD *)a3 + 48LL))(a3, v7 + 2, *v7) )
  {
LABEL_10:
    if ( !a3[173] )
      sub_16BD130("Symbol name with unsupported characters", 1u);
    v8 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v8 >= *(_QWORD *)(a2 + 16) )
    {
      sub_16E7DE0(a2, 34);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = v8 + 1;
      *v8 = 34;
    }
    v9 = &v5[v6];
    for ( i = *(_WORD **)(a2 + 24); v9 != v5; ++v5 )
    {
      v11 = *v5;
      v12 = *(_QWORD *)(a2 + 16);
      if ( *v5 == 10 )
      {
        if ( v12 - (unsigned __int64)i <= 1 )
        {
          sub_16E7EE0(a2, "\\n", 2u);
          i = *(_WORD **)(a2 + 24);
        }
        else
        {
          *i = 28252;
          i = (_WORD *)(*(_QWORD *)(a2 + 24) + 2LL);
          *(_QWORD *)(a2 + 24) = i;
        }
      }
      else if ( v11 == 34 )
      {
        if ( v12 - (unsigned __int64)i <= 1 )
        {
          sub_16E7EE0(a2, "\\\"", 2u);
          i = *(_WORD **)(a2 + 24);
        }
        else
        {
          *i = 8796;
          i = (_WORD *)(*(_QWORD *)(a2 + 24) + 2LL);
          *(_QWORD *)(a2 + 24) = i;
        }
      }
      else
      {
        if ( v12 <= (unsigned __int64)i )
        {
          sub_16E7DE0(a2, v11);
        }
        else
        {
          *(_QWORD *)(a2 + 24) = (char *)i + 1;
          *(_BYTE *)i = v11;
        }
        i = *(_WORD **)(a2 + 24);
      }
    }
    if ( (unsigned __int64)i >= *(_QWORD *)(a2 + 16) )
    {
      sub_16E7DE0(a2, 34);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = (char *)i + 1;
      *(_BYTE *)i = 34;
    }
    return;
  }
  v13 = *(void **)(a2 + 24);
  if ( v6 <= *(_QWORD *)(a2 + 16) - (_QWORD)v13 )
  {
    if ( v6 )
    {
      memcpy(v13, v7 + 2, v6);
      *(_QWORD *)(a2 + 24) += v6;
    }
  }
  else
  {
    sub_16E7EE0(a2, (char *)v7 + 16, v6);
  }
}
