// Function: sub_5E6C40
// Address: 0x5e6c40
//
__int64 __fastcall sub_5E6C40(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r15
  __int64 result; // rax
  _QWORD *v5; // rdx
  char v6; // al
  __int64 v7; // r14
  _BOOL4 v8; // r13d
  __int64 v9; // rbx
  __int64 v10; // r15
  unsigned __int8 v11; // al
  char v12; // [rsp+3h] [rbp-3Dh]
  unsigned int v13; // [rsp+4h] [rbp-3Ch]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = *(_BYTE *)(a1 + 80);
  if ( v2 != 16 )
  {
    v3 = a1;
    if ( v2 == 24 )
      v3 = *(_QWORD *)(a1 + 88);
LABEL_4:
    result = sub_878440();
    *(_QWORD *)(result + 8) = a1;
    if ( *(_BYTE *)(v3 + 80) == 20 )
    {
      *(_QWORD *)result = *(_QWORD *)(a2 + 48);
      *(_QWORD *)(a2 + 48) = result;
    }
    else
    {
      *(_QWORD *)result = *(_QWORD *)(a2 + 40);
      *(_QWORD *)(a2 + 40) = result;
    }
    return result;
  }
  v5 = *(_QWORD **)(a1 + 88);
  v3 = *v5;
  v6 = *(_BYTE *)(*v5 + 80LL);
  if ( v6 == 24 )
  {
    v3 = *(_QWORD *)(v3 + 88);
    v6 = *(_BYTE *)(v3 + 80);
  }
  if ( v6 != 17 )
    goto LABEL_4;
  v7 = v5[1];
  v14 = *(_QWORD *)(a1 + 64);
  if ( (*(_BYTE *)(v7 + 96) & 2) != 0 )
    result = sub_72B780(v7);
  else
    result = *(_QWORD *)(v7 + 112);
  v8 = (*(_BYTE *)(a1 + 82) & 4) != 0;
  v9 = *(_QWORD *)(v3 + 88);
  if ( v9 )
  {
    v13 = *(unsigned __int8 *)(result + 25);
    v12 = 8 * ((*(_BYTE *)(a1 + 96) & 0xC) != 0);
    do
    {
      v10 = sub_87F190(v9, v14, v7, 0, v8);
      v11 = sub_87D550(v9);
      *(_BYTE *)(v10 + 96) = *(_BYTE *)(v10 + 96) & 0xF4 | v12 | sub_87D600(v11, v13) & 3;
      result = sub_5E6C40(v10, a2);
      v9 = *(_QWORD *)(v9 + 8);
    }
    while ( v9 );
  }
  return result;
}
