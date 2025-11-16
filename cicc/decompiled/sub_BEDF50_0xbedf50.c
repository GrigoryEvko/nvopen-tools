// Function: sub_BEDF50
// Address: 0xbedf50
//
void __fastcall sub_BEDF50(__int64 *a1, __int64 a2, _BYTE **a3, const char **a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  _BYTE *v9; // rsi
  _BYTE *v10; // rdi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax

  v4 = *a1;
  if ( !*a1 )
  {
    *((_BYTE *)a1 + 152) = 1;
    return;
  }
  sub_CA0E80(a2, v4);
  v8 = *(_BYTE **)(v4 + 32);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v4 + 24) )
  {
    sub_CB5D20(v4, 10);
  }
  else
  {
    *(_QWORD *)(v4 + 32) = v8 + 1;
    *v8 = 10;
  }
  v9 = (_BYTE *)*a1;
  *((_BYTE *)a1 + 152) = 1;
  if ( v9 )
  {
    v10 = *a3;
    if ( !*a3 )
      goto LABEL_9;
    if ( *v10 > 0x1Cu )
    {
      sub_A693B0((__int64)v10, v9, (__int64)(a1 + 2), 0);
      v11 = *a1;
      v12 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*a1 + 24) )
        goto LABEL_8;
    }
    else
    {
      sub_A5C020(v10, (__int64)v9, 1, (__int64)(a1 + 2));
      v11 = *a1;
      v12 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*a1 + 24) )
      {
LABEL_8:
        *(_QWORD *)(v11 + 32) = v12 + 1;
        *v12 = 10;
        goto LABEL_9;
      }
    }
    sub_CB5D20(v11, 10);
LABEL_9:
    if ( *a4 )
    {
      sub_A62C00(*a4, *a1, (__int64)(a1 + 2), a1[1]);
      v13 = *a1;
      v14 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(*a1 + 24) )
      {
        sub_CB5D20(v13, 10);
      }
      else
      {
        *(_QWORD *)(v13 + 32) = v14 + 1;
        *v14 = 10;
      }
    }
  }
}
