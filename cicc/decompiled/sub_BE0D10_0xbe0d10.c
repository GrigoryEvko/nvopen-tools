// Function: sub_BE0D10
// Address: 0xbe0d10
//
void __fastcall sub_BE0D10(_BYTE *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v5; // al
  __int64 v6; // rax
  const char *v7; // rax
  __int64 v8; // r14
  _BYTE *v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rax
  _BYTE *v14; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v15[4]; // [rsp+10h] [rbp-50h] BYREF
  char v16; // [rsp+30h] [rbp-30h]
  char v17; // [rsp+31h] [rbp-2Fh]

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 14 )
  {
    if ( ((*(_BYTE *)a2 - 61) & 0xEF) != 0 )
    {
      v14 = (_BYTE *)a2;
      v7 = "dereferenceable, dereferenceable_or_null apply only to load and inttoptr instructions, use attributes for calls or invokes";
      v17 = 1;
      goto LABEL_20;
    }
    v5 = *(_BYTE *)(a3 - 16);
    if ( (v5 & 2) != 0 )
    {
      if ( *(_DWORD *)(a3 - 24) == 1 )
      {
        v6 = **(_QWORD **)(a3 - 32);
        if ( *(_BYTE *)v6 != 1 )
        {
LABEL_6:
          v14 = (_BYTE *)a2;
          v7 = "dereferenceable, dereferenceable_or_null metadata value must be an i64!";
          v17 = 1;
LABEL_20:
          v15[0] = v7;
          v16 = 3;
          sub_BE0C10(a1, (__int64)v15, &v14);
          return;
        }
        goto LABEL_16;
      }
    }
    else if ( ((*(_WORD *)(a3 - 16) >> 6) & 0xF) == 1 )
    {
      v6 = *(_QWORD *)(a3 - 8LL * ((v5 >> 2) & 0xF) - 16);
      if ( *(_BYTE *)v6 != 1 )
        goto LABEL_6;
LABEL_16:
      v13 = *(_QWORD *)(v6 + 136);
      if ( *(_BYTE *)v13 != 17 || !sub_BCAC40(*(_QWORD *)(v13 + 8), 64) )
        goto LABEL_6;
      return;
    }
    v14 = (_BYTE *)a2;
    v7 = "dereferenceable, dereferenceable_or_null take one operand!";
    v17 = 1;
    goto LABEL_20;
  }
  v8 = *(_QWORD *)a1;
  v17 = 1;
  v15[0] = "dereferenceable, dereferenceable_or_null apply only to pointer types";
  v16 = 3;
  if ( !v8 )
  {
    a1[152] = 1;
    return;
  }
  sub_CA0E80(v15, v8);
  v9 = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
  {
    sub_CB5D20(v8, 10);
  }
  else
  {
    *(_QWORD *)(v8 + 32) = v9 + 1;
    *v9 = 10;
  }
  v10 = *(_BYTE **)a1;
  a1[152] = 1;
  if ( v10 )
  {
    if ( *(_BYTE *)a2 <= 0x1Cu )
    {
      sub_A5C020((_BYTE *)a2, (__int64)v10, 1, (__int64)(a1 + 16));
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_13;
    }
    else
    {
      sub_A693B0(a2, v10, (__int64)(a1 + 16), 0);
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_13:
        *(_QWORD *)(v11 + 32) = v12 + 1;
        *v12 = 10;
        return;
      }
    }
    sub_CB5D20(v11, 10);
  }
}
