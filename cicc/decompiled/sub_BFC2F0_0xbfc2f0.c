// Function: sub_BFC2F0
// Address: 0xbfc2f0
//
void __fastcall sub_BFC2F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // ecx
  char v4; // dl
  const char *v5; // rax
  __int64 v6; // r14
  _BYTE *v7; // rax
  __int64 v8; // rax
  const char *v9; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-30h]
  char v11; // [rsp+21h] [rbp-2Fh]

  v2 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  if ( *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL) != v2 )
  {
    v11 = 1;
    v5 = "Both operands to ICmp instruction are not of the same type!";
    goto LABEL_13;
  }
  v3 = *(unsigned __int8 *)(v2 + 8);
  v4 = *(_BYTE *)(v2 + 8);
  if ( (unsigned int)(v3 - 17) > 1 )
  {
    if ( (_BYTE)v3 == 12 )
      goto LABEL_7;
  }
  else
  {
    if ( *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL) == 12 )
      goto LABEL_7;
    if ( v3 == 18 )
    {
      v4 = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
      goto LABEL_6;
    }
  }
  if ( v3 == 17 )
    v4 = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
LABEL_6:
  if ( v4 != 14 )
  {
    v11 = 1;
    v5 = "Invalid operand types for ICmp instruction";
    goto LABEL_13;
  }
LABEL_7:
  if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 9 )
  {
    sub_BF6FE0(a1, a2);
    return;
  }
  v11 = 1;
  v5 = "Invalid predicate in ICmp instruction!";
LABEL_13:
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
    *(_BYTE *)(a1 + 152) = 1;
    if ( v8 )
      sub_BDBD80(a1, (_BYTE *)a2);
  }
  else
  {
    *(_BYTE *)(a1 + 152) = 1;
  }
}
