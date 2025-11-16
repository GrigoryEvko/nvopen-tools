// Function: sub_1595F50
// Address: 0x1595f50
//
char __fastcall sub_1595F50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v5; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  _QWORD *v13; // rax
  char v14; // bl
  _BYTE v15[8]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 != 14 )
  {
    if ( v4 == 12 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(sub_1595890(a1) + 8) - 1) <= 5u && (unsigned __int8)sub_1595CF0(a1) )
      {
        a2 = a1;
        sub_1595B70((__int64)v15, a1, 0);
        v12 = sub_16982C0(v15, a1, v10, v11);
        v13 = v16;
        if ( v16[0] == v12 )
          v13 = (_QWORD *)(v16[1] + 8LL);
        v14 = *((_BYTE *)v13 + 18) & 7;
        sub_127D120(v16);
        if ( v14 == 3 )
          return 1;
      }
      v4 = *(_BYTE *)(a1 + 16);
    }
    if ( v4 != 8 )
      return sub_1593BB0(a1, a2, a3, a4);
    v7 = sub_1594B20(a1);
    v8 = v7;
    if ( !v7 || *(_BYTE *)(v7 + 16) != 14 )
      return sub_1593BB0(a1, a2, a3, a4);
    v9 = *(_QWORD *)(v7 + 32) == sub_16982C0(a1, a2, a3, a4) ? *(_QWORD *)(v8 + 40) + 8LL : v8 + 32;
    if ( (*(_BYTE *)(v9 + 18) & 7) != 3 )
      return sub_1593BB0(a1, a2, a3, a4);
    return 1;
  }
  if ( *(_QWORD *)(a1 + 32) == sub_16982C0(a1, a2, a3, a4) )
    v5 = *(_QWORD *)(a1 + 40) + 8LL;
  else
    v5 = a1 + 32;
  return (*(_BYTE *)(v5 + 18) & 7) == 3;
}
