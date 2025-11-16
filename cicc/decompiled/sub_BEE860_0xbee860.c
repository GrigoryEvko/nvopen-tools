// Function: sub_BEE860
// Address: 0xbee860
//
void __fastcall sub_BEE860(_BYTE *a1, __int64 a2, _BYTE **a3, _BYTE *a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  _BYTE *v9; // rsi
  _BYTE *v10; // rdi
  __int64 v11; // rdi
  _BYTE *v12; // rax
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
    v10 = *a3;
    if ( !*a3 )
      goto LABEL_9;
    if ( *v10 > 0x1Cu )
    {
      sub_A693B0((__int64)v10, v9, (__int64)(a1 + 16), 0);
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_8;
    }
    else
    {
      sub_A5C020(v10, (__int64)v9, 1, (__int64)(a1 + 16));
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_8:
        *(_QWORD *)(v11 + 32) = v12 + 1;
        *v12 = 10;
        v9 = *(_BYTE **)a1;
        goto LABEL_9;
      }
    }
    sub_CB5D20(v11, 10);
    v9 = *(_BYTE **)a1;
LABEL_9:
    if ( *a4 <= 0x1Cu )
    {
      sub_A5C020(a4, (__int64)v9, 1, (__int64)(a1 + 16));
      v13 = *(_QWORD *)a1;
      v14 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v14 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_11;
    }
    else
    {
      sub_A693B0((__int64)a4, v9, (__int64)(a1 + 16), 0);
      v13 = *(_QWORD *)a1;
      v14 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v14 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_11:
        *(_QWORD *)(v13 + 32) = v14 + 1;
        *v14 = 10;
        return;
      }
    }
    sub_CB5D20(v13, 10);
  }
}
