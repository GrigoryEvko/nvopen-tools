// Function: sub_165A340
// Address: 0x165a340
//
void __fastcall sub_165A340(_BYTE *a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rdi
  _BYTE *v16; // rax

  v4 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    a1[72] = 1;
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
  v9 = *(_QWORD *)a1;
  a1[72] = 1;
  if ( v9 )
  {
    v10 = *a3;
    if ( !*a3 )
      goto LABEL_9;
    if ( *(_BYTE *)(v10 + 16) <= 0x17u )
    {
      sub_1553920((__int64 *)v10, v9, 1, (__int64)(a1 + 16));
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        goto LABEL_8;
    }
    else
    {
      sub_155BD40(v10, v9, (__int64)(a1 + 16), 0);
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
LABEL_8:
        *(_QWORD *)(v11 + 24) = v12 + 1;
        *v12 = 10;
        goto LABEL_9;
      }
    }
    sub_16E7DE0(v11, 10);
LABEL_9:
    v13 = *a4;
    if ( !*a4 )
      return;
    v14 = *(_QWORD *)a1;
    if ( *(_BYTE *)(v13 + 16) <= 0x17u )
    {
      sub_1553920((__int64 *)v13, v14, 1, (__int64)(a1 + 16));
      v15 = *(_QWORD *)a1;
      v16 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v16 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        goto LABEL_12;
    }
    else
    {
      sub_155BD40(v13, v14, (__int64)(a1 + 16), 0);
      v15 = *(_QWORD *)a1;
      v16 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v16 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
LABEL_12:
        *(_QWORD *)(v15 + 24) = v16 + 1;
        *v16 = 10;
        return;
      }
    }
    sub_16E7DE0(v15, 10);
  }
}
