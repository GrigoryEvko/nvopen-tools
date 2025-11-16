// Function: sub_165C320
// Address: 0x165c320
//
void __fastcall sub_165C320(__int64 *a1, __int64 a2, __int64 *a3, unsigned __int8 **a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax

  v4 = *a1;
  if ( !*a1 )
  {
    *((_BYTE *)a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(a2, v4);
  v8 = *(_BYTE **)(v4 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v4 + 16) )
  {
    sub_16E7DE0(v4, 10);
  }
  else
  {
    *(_QWORD *)(v4 + 24) = v8 + 1;
    *v8 = 10;
  }
  v9 = *a1;
  *((_BYTE *)a1 + 72) = 1;
  if ( v9 )
  {
    v10 = *a3;
    if ( !*a3 )
      goto LABEL_9;
    if ( *(_BYTE *)(v10 + 16) <= 0x17u )
    {
      sub_1553920((__int64 *)v10, v9, 1, (__int64)(a1 + 2));
      v11 = *a1;
      v12 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*a1 + 16) )
        goto LABEL_8;
    }
    else
    {
      sub_155BD40(v10, v9, (__int64)(a1 + 2), 0);
      v11 = *a1;
      v12 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*a1 + 16) )
      {
LABEL_8:
        *(_QWORD *)(v11 + 24) = v12 + 1;
        *v12 = 10;
        goto LABEL_9;
      }
    }
    sub_16E7DE0(v11, 10);
LABEL_9:
    if ( *a4 )
    {
      sub_15562E0(*a4, *a1, (__int64)(a1 + 2), a1[1]);
      v13 = *a1;
      v14 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(*a1 + 16) )
      {
        sub_16E7DE0(v13, 10);
      }
      else
      {
        *(_QWORD *)(v13 + 24) = v14 + 1;
        *v14 = 10;
      }
    }
  }
}
