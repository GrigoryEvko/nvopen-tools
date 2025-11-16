// Function: sub_8C7F70
// Address: 0x8c7f70
//
__int64 __fastcall sub_8C7F70(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int8 v3; // al
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  bool v11; // cl
  __int64 v12; // rdx
  bool v13; // r9
  char v14[49]; // [rsp+Fh] [rbp-31h] BYREF

  v2 = 0;
  v3 = *(_BYTE *)(a1 + 81);
  if ( ((v3 ^ *(_BYTE *)(a2 + 81)) & 0x10) != 0 )
    return v2;
  if ( (v3 & 0x10) != 0 )
  {
    v4 = *(_QWORD *)(a1 + 64);
    v5 = *(_QWORD *)(a2 + 64);
    v2 = 1;
    if ( v4 == v5 )
      return v2;
    goto LABEL_4;
  }
  v7 = sub_87D1A0(a1, v14);
  v8 = sub_87D1A0(a2, v14);
  if ( v8 != 0 && v7 != 0 )
  {
    if ( ((*(_BYTE *)(v7 + 89) ^ *(_BYTE *)(v8 + 89)) & 4) != 0 )
      return v2;
    if ( (*(_BYTE *)(v7 + 89) & 4) != 0 )
    {
      v4 = *(_QWORD *)(a1 + 64);
      v5 = *(_QWORD *)(a2 + 64);
      v2 = 1;
      if ( v5 == v4 )
        return v2;
LABEL_4:
      v2 = 0;
      if ( *qword_4D03FD0 )
      {
        if ( v4 && v5 )
          return (unsigned int)sub_8C7EB0(v4, v5, 6u) != 0;
        return 0;
      }
      return v2;
    }
  }
  v9 = *(_QWORD *)(a1 + 64);
  v10 = *(_QWORD *)(a2 + 64);
  if ( v9 || !v7 )
    goto LABEL_18;
  v9 = *(_QWORD *)(v7 + 40);
  v11 = 0;
  if ( v9 )
  {
    if ( *(_BYTE *)(v9 + 28) != 3 )
    {
      v9 = 0;
      goto LABEL_19;
    }
    v9 = *(_QWORD *)(v9 + 32);
LABEL_18:
    v11 = v9 != 0;
  }
LABEL_19:
  if ( v10 || !v8 )
  {
LABEL_25:
    v12 = v10 | v9;
    v13 = v10 != 0;
    goto LABEL_26;
  }
  v10 = *(_QWORD *)(v8 + 40);
  if ( v10 )
  {
    if ( *(_BYTE *)(v10 + 28) != 3 )
    {
      v12 = v9;
      v13 = 0;
      v10 = 0;
      goto LABEL_26;
    }
    v10 = *(_QWORD *)(v10 + 32);
    goto LABEL_25;
  }
  v12 = v9;
  v13 = 0;
LABEL_26:
  v2 = 1;
  if ( v12 && (v8 == 0 || v7 == 0 || (*(_BYTE *)(v7 + 88) & 0x70) != 0x30 || (*(_BYTE *)(v8 + 88) & 0x70) != 0x30) )
  {
    v2 = 1;
    if ( v9 != v10 )
    {
      v2 = 0;
      if ( *qword_4D03FD0 )
      {
        if ( v13 && v11 )
          return (unsigned int)sub_8C7EB0(v9, v10, 0x1Cu) != 0;
        return 0;
      }
    }
  }
  return v2;
}
