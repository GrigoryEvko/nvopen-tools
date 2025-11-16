// Function: sub_BF90E0
// Address: 0xbf90e0
//
void __fastcall sub_BF90E0(_BYTE *a1, __int64 a2)
{
  __int64 v3; // r13
  unsigned __int64 v4; // rax
  __int64 v5; // r14
  _BYTE *v6; // rax
  _BYTE *v7; // rsi
  __int64 v8; // rdi
  _BYTE *v9; // rax
  const char *v10; // [rsp+0h] [rbp-50h] BYREF
  char v11; // [rsp+20h] [rbp-30h]
  char v12; // [rsp+21h] [rbp-2Fh]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 != v3 + 48 )
  {
    if ( !v4 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 <= 0xA && a2 == v4 - 24 )
    {
      sub_BF6FE0((__int64)a1, a2);
      return;
    }
  }
  v5 = *(_QWORD *)a1;
  v12 = 1;
  v10 = "Terminator found in the middle of a basic block!";
  v11 = 3;
  if ( v5 )
  {
    sub_CA0E80(&v10, v5);
    v6 = *(_BYTE **)(v5 + 32);
    if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 24) )
    {
      sub_CB5D20(v5, 10);
    }
    else
    {
      *(_QWORD *)(v5 + 32) = v6 + 1;
      *v6 = 10;
    }
    v7 = *(_BYTE **)a1;
    a1[152] = 1;
    if ( v7 )
    {
      if ( *(_BYTE *)v3 <= 0x1Cu )
      {
        sub_A5C020((_BYTE *)v3, (__int64)v7, 1, (__int64)(a1 + 16));
        v8 = *(_QWORD *)a1;
        v9 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
        if ( (unsigned __int64)v9 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          goto LABEL_12;
      }
      else
      {
        sub_A693B0(v3, v7, (__int64)(a1 + 16), 0);
        v8 = *(_QWORD *)a1;
        v9 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
        if ( (unsigned __int64)v9 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        {
LABEL_12:
          *(_QWORD *)(v8 + 32) = v9 + 1;
          *v9 = 10;
          return;
        }
      }
      sub_CB5D20(v8, 10);
    }
  }
  else
  {
    a1[152] = 1;
  }
}
