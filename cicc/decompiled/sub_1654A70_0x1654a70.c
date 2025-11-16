// Function: sub_1654A70
// Address: 0x1654a70
//
void __fastcall sub_1654A70(_BYTE *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  const char *v7; // rax
  __int64 v8; // r14
  _BYTE *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-40h] BYREF
  char v15; // [rsp+20h] [rbp-30h]
  char v16; // [rsp+21h] [rbp-2Fh]

  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 )
  {
    v8 = *(_QWORD *)a1;
    v16 = 1;
    v14[0] = "dereferenceable, dereferenceable_or_null apply only to pointer types";
    v15 = 3;
    if ( !v8 )
    {
      a1[72] = 1;
      return;
    }
    sub_16E2CE0(v14, v8);
    v9 = *(_BYTE **)(v8 + 24);
    if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
    {
      sub_16E7DE0(v8, 10);
    }
    else
    {
      *(_QWORD *)(v8 + 24) = v9 + 1;
      *v9 = 10;
    }
    v10 = *(_QWORD *)a1;
    a1[72] = 1;
    if ( !v10 )
      return;
    if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    {
      sub_1553920((__int64 *)a2, v10, 1, (__int64)(a1 + 16));
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        goto LABEL_17;
    }
    else
    {
      sub_155BD40(a2, v10, (__int64)(a1 + 16), 0);
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
LABEL_17:
        *(_QWORD *)(v11 + 24) = v12 + 1;
        *v12 = 10;
        return;
      }
    }
    sub_16E7DE0(v11, 10);
    return;
  }
  if ( *(_BYTE *)(a2 + 16) != 54 )
  {
    v13 = a2;
    v7 = "dereferenceable, dereferenceable_or_null apply only to load instructions, use attributes for calls or invokes";
    v16 = 1;
    goto LABEL_9;
  }
  if ( *(_DWORD *)(a3 + 8) != 1 )
  {
    v13 = a2;
    v7 = "dereferenceable, dereferenceable_or_null take one operand!";
    v16 = 1;
    goto LABEL_9;
  }
  v5 = *(_QWORD *)(a3 - 8);
  if ( *(_BYTE *)v5 != 1 || (v6 = *(_QWORD *)(v5 + 136), *(_BYTE *)(v6 + 16) != 13) || !sub_1642F90(*(_QWORD *)v6, 64) )
  {
    v13 = a2;
    v7 = "dereferenceable, dereferenceable_or_null metadata value must be an i64!";
    v16 = 1;
LABEL_9:
    v14[0] = v7;
    v15 = 3;
    sub_1654980(a1, (__int64)v14, &v13);
  }
}
