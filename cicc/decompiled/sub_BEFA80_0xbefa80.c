// Function: sub_BEFA80
// Address: 0xbefa80
//
void __fastcall sub_BEFA80(_BYTE *a1, __int64 a2, _BYTE *a3, _BYTE **a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // rdi
  _BYTE *v11; // rax
  _BYTE *v12; // rdi
  _BYTE *v13; // rsi
  __int64 v14; // rdi
  _BYTE *v15; // rax

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
    v12 = *a4;
    if ( !*a4 )
      return;
    v13 = *(_BYTE **)a1;
    if ( *v12 > 0x1Cu )
    {
      sub_A693B0((__int64)v12, v13, (__int64)(a1 + 16), 0);
      v14 = *(_QWORD *)a1;
      v15 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v15 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_11;
    }
    else
    {
      sub_A5C020(v12, (__int64)v13, 1, (__int64)(a1 + 16));
      v14 = *(_QWORD *)a1;
      v15 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v15 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_11:
        *(_QWORD *)(v14 + 32) = v15 + 1;
        *v15 = 10;
        return;
      }
    }
    sub_CB5D20(v14, 10);
  }
}
