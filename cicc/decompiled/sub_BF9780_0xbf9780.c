// Function: sub_BF9780
// Address: 0xbf9780
//
void __fastcall sub_BF9780(_BYTE *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  const char *v5; // rax
  __int64 v6; // r14
  _BYTE *v7; // rax
  __int64 v8; // rax
  const char *v9; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-30h]
  char v11; // [rsp+21h] [rbp-2Fh]

  v3 = *(_QWORD *)(a2 - 8);
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v3 + 8LL) + 8LL) == 14 )
  {
    LODWORD(v4) = 0;
    do
    {
      if ( (_DWORD)v4 == (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) - 1 )
      {
        sub_BF90E0(a1, a2);
        return;
      }
      v4 = (unsigned int)(v4 + 1);
    }
    while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v3 + 32 * v4) + 8LL) + 8LL) == 8 );
    v11 = 1;
    v5 = "Indirectbr destinations must all have pointer type!";
  }
  else
  {
    v11 = 1;
    v5 = "Indirectbr operand must have pointer type!";
  }
  v6 = *(_QWORD *)a1;
  v9 = v5;
  v10 = 3;
  if ( v6 )
  {
    sub_CA0E80(&v9, v6);
    v7 = *(_BYTE **)(v6 + 32);
    if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 24) )
    {
      sub_CB5D20(v6, 10);
    }
    else
    {
      *(_QWORD *)(v6 + 32) = v7 + 1;
      *v7 = 10;
    }
    v8 = *(_QWORD *)a1;
    a1[152] = 1;
    if ( v8 )
      sub_BDBD80((__int64)a1, (_BYTE *)a2);
  }
  else
  {
    a1[152] = 1;
  }
}
