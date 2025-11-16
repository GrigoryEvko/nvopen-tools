// Function: sub_38D0580
// Address: 0x38d0580
//
void __fastcall sub_38D0580(__int64 a1, __int64 a2, char *a3, size_t a4)
{
  __int64 v7; // r15
  void *v8; // rdi
  __int64 v9; // rdi
  _BYTE *v10; // rax
  _BYTE *v11; // rax
  __int64 v12; // r13
  _BYTE *v13; // rdi
  __int64 v14; // rax
  char *v15; // rsi
  size_t v16; // r15
  _BYTE *v17; // rax
  void *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax

  if ( a4 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( v7 )
    {
      v8 = *(void **)(v7 + 24);
      if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 < a4 )
      {
        sub_16E7EE0(v7, a3, a4);
        if ( a3[a4 - 1] == 10 )
          return;
      }
      else
      {
        memcpy(v8, a3, a4);
        *(_QWORD *)(v7 + 24) += a4;
        if ( a3[a4 - 1] == 10 )
          return;
      }
      v9 = *(_QWORD *)(a1 + 8);
      v10 = *(_BYTE **)(v9 + 24);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
      {
        sub_16E7DE0(v9, 10);
      }
      else
      {
        *(_QWORD *)(v9 + 24) = v10 + 1;
        *v10 = 10;
      }
    }
    else
    {
      v11 = *(_BYTE **)(a2 + 24);
      v12 = a2;
      if ( *(_BYTE **)(a2 + 16) == v11 )
      {
        v19 = sub_16E7EE0(a2, " ", 1u);
        v13 = *(_BYTE **)(v19 + 24);
        v12 = v19;
      }
      else
      {
        *v11 = 32;
        v13 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 1LL);
        *(_QWORD *)(a2 + 24) = v13;
      }
      v14 = *(_QWORD *)(a1 + 16);
      v15 = *(char **)(v14 + 48);
      v16 = *(_QWORD *)(v14 + 56);
      v17 = *(_BYTE **)(v12 + 16);
      if ( v16 > v17 - v13 )
      {
        v12 = sub_16E7EE0(v12, v15, v16);
        v17 = *(_BYTE **)(v12 + 16);
        v13 = *(_BYTE **)(v12 + 24);
      }
      else if ( v16 )
      {
        memcpy(v13, v15, v16);
        v17 = *(_BYTE **)(v12 + 16);
        v13 = (_BYTE *)(v16 + *(_QWORD *)(v12 + 24));
        *(_QWORD *)(v12 + 24) = v13;
      }
      if ( v13 == v17 )
      {
        v20 = sub_16E7EE0(v12, " ", 1u);
        v18 = *(void **)(v20 + 24);
        v12 = v20;
      }
      else
      {
        *v13 = 32;
        v18 = (void *)(*(_QWORD *)(v12 + 24) + 1LL);
        *(_QWORD *)(v12 + 24) = v18;
      }
      if ( *(_QWORD *)(v12 + 16) - (_QWORD)v18 < a4 )
      {
        sub_16E7EE0(v12, a3, a4);
      }
      else
      {
        memcpy(v18, a3, a4);
        *(_QWORD *)(v12 + 24) += a4;
      }
    }
  }
}
