// Function: sub_BEFED0
// Address: 0xbefed0
//
void __fastcall sub_BEFED0(_BYTE *a1, __int64 a2, _BYTE **a3, _BYTE **a4, _BYTE **a5)
{
  __int64 v6; // r12
  _BYTE *v10; // rax
  _BYTE *v11; // rsi
  _BYTE *v12; // rdi
  __int64 v13; // rdi
  _BYTE *v14; // rax
  _BYTE *v15; // rdi
  _BYTE *v16; // rsi
  __int64 v17; // rdi
  _BYTE *v18; // rax
  _BYTE *v19; // rdi
  _BYTE *v20; // rsi
  __int64 v21; // rdi
  _BYTE *v22; // rax

  v6 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    a1[152] = 1;
    return;
  }
  sub_CA0E80(a2, v6);
  v10 = *(_BYTE **)(v6 + 32);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v6 + 24) )
  {
    sub_CB5D20(v6, 10);
  }
  else
  {
    *(_QWORD *)(v6 + 32) = v10 + 1;
    *v10 = 10;
  }
  v11 = *(_BYTE **)a1;
  a1[152] = 1;
  if ( v11 )
  {
    v12 = *a3;
    if ( !*a3 )
      goto LABEL_9;
    if ( *v12 > 0x1Cu )
    {
      sub_A693B0((__int64)v12, v11, (__int64)(a1 + 16), 0);
      v13 = *(_QWORD *)a1;
      v14 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v14 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_8;
    }
    else
    {
      sub_A5C020(v12, (__int64)v11, 1, (__int64)(a1 + 16));
      v13 = *(_QWORD *)a1;
      v14 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v14 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_8:
        *(_QWORD *)(v13 + 32) = v14 + 1;
        *v14 = 10;
        goto LABEL_9;
      }
    }
    sub_CB5D20(v13, 10);
LABEL_9:
    v15 = *a4;
    if ( !*a4 )
      goto LABEL_13;
    v16 = *(_BYTE **)a1;
    if ( *v15 > 0x1Cu )
    {
      sub_A693B0((__int64)v15, v16, (__int64)(a1 + 16), 0);
      v17 = *(_QWORD *)a1;
      v18 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v18 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_12;
    }
    else
    {
      sub_A5C020(v15, (__int64)v16, 1, (__int64)(a1 + 16));
      v17 = *(_QWORD *)a1;
      v18 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v18 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_12:
        *(_QWORD *)(v17 + 32) = v18 + 1;
        *v18 = 10;
        goto LABEL_13;
      }
    }
    sub_CB5D20(v17, 10);
LABEL_13:
    v19 = *a5;
    if ( !*a5 )
      return;
    v20 = *(_BYTE **)a1;
    if ( *v19 > 0x1Cu )
    {
      sub_A693B0((__int64)v19, v20, (__int64)(a1 + 16), 0);
      v21 = *(_QWORD *)a1;
      v22 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v22 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_16;
    }
    else
    {
      sub_A5C020(v19, (__int64)v20, 1, (__int64)(a1 + 16));
      v21 = *(_QWORD *)a1;
      v22 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v22 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_16:
        *(_QWORD *)(v21 + 32) = v22 + 1;
        *v22 = 10;
        return;
      }
    }
    sub_CB5D20(v21, 10);
  }
}
