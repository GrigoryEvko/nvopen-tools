// Function: sub_BEF920
// Address: 0xbef920
//
void __fastcall sub_BEF920(_BYTE *a1, __int64 a2, _BYTE *a3, _BYTE *a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // rdi
  _BYTE *v11; // rax
  _BYTE *v12; // rsi
  __int64 v13; // rdi
  _BYTE *v14; // rax

  v4 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    a1[152] = 1;
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
  v9 = *(_BYTE **)a1;
  a1[152] = 1;
  if ( v9 )
  {
    if ( *a3 <= 0x1Cu )
    {
      sub_A5C020(a3, (__int64)v9, 1, (__int64)(a1 + 16));
      v10 = *(_QWORD *)a1;
      v11 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v11 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_7;
    }
    else
    {
      sub_A693B0((__int64)a3, v9, (__int64)(a1 + 16), 0);
      v10 = *(_QWORD *)a1;
      v11 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v11 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_7:
        *(_QWORD *)(v10 + 32) = v11 + 1;
        *v11 = 10;
        goto LABEL_8;
      }
    }
    sub_CB5D20(v10, 10);
LABEL_8:
    v12 = *(_BYTE **)a1;
    if ( *a4 <= 0x1Cu )
    {
      sub_A5C020(a4, (__int64)v12, 1, (__int64)(a1 + 16));
      v13 = *(_QWORD *)a1;
      v14 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v14 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_10;
    }
    else
    {
      sub_A693B0((__int64)a4, v12, (__int64)(a1 + 16), 0);
      v13 = *(_QWORD *)a1;
      v14 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v14 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_10:
        *(_QWORD *)(v13 + 32) = v14 + 1;
        *v14 = 10;
        return;
      }
    }
    sub_CB5D20(v13, 10);
  }
}
