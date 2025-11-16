// Function: sub_165BF10
// Address: 0x165bf10
//
void __fastcall sub_165BF10(_BYTE *a1, __int64 a2, _QWORD *a3, __int64 *a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  unsigned __int8 v11; // al
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdi
  _BYTE *v17; // rax

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
    v10 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
    v11 = *(_BYTE *)(v10 + 16);
    if ( v11 > 0x17u && (v11 == 78 || v11 == 29) )
    {
      if ( v10 )
      {
        sub_155BD40(v10, v9, (__int64)(a1 + 16), 0);
        v16 = *(_QWORD *)a1;
        v17 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v17 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
          sub_16E7DE0(v16, 10);
        }
        else
        {
          *(_QWORD *)(v16 + 24) = v17 + 1;
          *v17 = 10;
        }
      }
    }
    v12 = *a4;
    if ( *a4 )
    {
      v13 = *(_QWORD *)a1;
      if ( *(_BYTE *)(v12 + 16) <= 0x17u )
      {
        sub_1553920((__int64 *)v12, v13, 1, (__int64)(a1 + 16));
        v14 = *(_QWORD *)a1;
        v15 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v15 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          goto LABEL_11;
      }
      else
      {
        sub_155BD40(v12, v13, (__int64)(a1 + 16), 0);
        v14 = *(_QWORD *)a1;
        v15 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v15 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
LABEL_11:
          *(_QWORD *)(v14 + 24) = v15 + 1;
          *v15 = 10;
          return;
        }
      }
      sub_16E7DE0(v14, 10);
    }
  }
}
