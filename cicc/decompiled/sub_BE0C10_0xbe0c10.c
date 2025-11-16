// Function: sub_BE0C10
// Address: 0xbe0c10
//
void __fastcall sub_BE0C10(_BYTE *a1, __int64 a2, _BYTE **a3)
{
  __int64 v4; // r12
  _BYTE *v6; // rax
  _BYTE *v7; // rsi
  _BYTE *v8; // rdi
  __int64 v9; // rdi
  _BYTE *v10; // rax

  v4 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    sub_CA0E80(a2, v4);
    v6 = *(_BYTE **)(v4 + 32);
    if ( (unsigned __int64)v6 >= *(_QWORD *)(v4 + 24) )
    {
      sub_CB5D20(v4, 10);
    }
    else
    {
      *(_QWORD *)(v4 + 32) = v6 + 1;
      *v6 = 10;
    }
    v7 = *(_BYTE **)a1;
    a1[152] = 1;
    if ( v7 )
    {
      v8 = *a3;
      if ( *a3 )
      {
        if ( *v8 <= 0x1Cu )
        {
          sub_A5C020(v8, (__int64)v7, 1, (__int64)(a1 + 16));
          v9 = *(_QWORD *)a1;
          v10 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
          if ( (unsigned __int64)v10 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            goto LABEL_8;
        }
        else
        {
          sub_A693B0((__int64)v8, v7, (__int64)(a1 + 16), 0);
          v9 = *(_QWORD *)a1;
          v10 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
          if ( (unsigned __int64)v10 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
LABEL_8:
            sub_CB5D20(v9, 10);
            return;
          }
        }
        *(_QWORD *)(v9 + 32) = v10 + 1;
        *v10 = 10;
      }
    }
  }
  else
  {
    a1[152] = 1;
  }
}
