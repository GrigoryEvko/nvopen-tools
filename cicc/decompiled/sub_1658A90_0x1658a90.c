// Function: sub_1658A90
// Address: 0x1658a90
//
void __fastcall sub_1658A90(_BYTE *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r12
  _BYTE *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rdi
  _BYTE *v10; // rax

  v4 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    a1[72] = 1;
    return;
  }
  sub_16E2CE0(a2, v4);
  v6 = *(_BYTE **)(v4 + 24);
  if ( (unsigned __int64)v6 >= *(_QWORD *)(v4 + 16) )
  {
    sub_16E7DE0(v4, 10);
  }
  else
  {
    *(_QWORD *)(v4 + 24) = v6 + 1;
    *v6 = 10;
  }
  v7 = *(_QWORD *)a1;
  a1[72] = 1;
  if ( v7 )
  {
    v8 = *a3;
    if ( *a3 )
    {
      if ( *(_BYTE *)(v8 + 16) <= 0x17u )
      {
        sub_1553920((__int64 *)v8, v7, 1, (__int64)(a1 + 16));
        v9 = *(_QWORD *)a1;
        v10 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v10 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          goto LABEL_8;
      }
      else
      {
        sub_155BD40(v8, v7, (__int64)(a1 + 16), 0);
        v9 = *(_QWORD *)a1;
        v10 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v10 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
LABEL_8:
          *(_QWORD *)(v9 + 24) = v10 + 1;
          *v10 = 10;
          return;
        }
      }
      sub_16E7DE0(v9, 10);
    }
  }
}
