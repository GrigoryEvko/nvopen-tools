// Function: sub_39C0690
// Address: 0x39c0690
//
__int64 __fastcall sub_39C0690(__int64 *a1, __int64 *a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // r14
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rcx
  signed __int64 v16; // r8
  char *v17; // r9
  __int64 v18; // [rsp+8h] [rbp-48h]
  char v19; // [rsp+17h] [rbp-39h]

  v3 = 0;
  v4 = a1[1];
  v5 = *a2;
  v6 = *(_QWORD *)(*(_QWORD *)(v4 + 256) + 16LL);
  if ( (*(_BYTE *)(*a2 + 18) & 8) != 0 )
  {
    v11 = sub_15E38F0(*a2);
    v3 = sub_1649C60(v11);
    if ( *(_BYTE *)(v3 + 16) )
      v3 = 0;
    if ( (*(_BYTE *)(v5 + 18) & 8) != 0
      && !(unsigned int)sub_14DD7D0(v3)
      && ((unsigned __int8)sub_1560180(v5 + 112, 56)
       || !(unsigned __int8)sub_1560180(v5 + 112, 30)
       || (*(_BYTE *)(v5 + 18) & 8) != 0) )
    {
      v8 = 56;
      v12 = **(_QWORD **)(a1[1] + 264);
      if ( (unsigned __int8)sub_1560180(v12 + 112, 56) )
        goto LABEL_12;
      v8 = 30;
      v19 = sub_1560180(v12 + 112, 30);
      if ( !v19 || (*(_BYTE *)(v12 + 18) & 8) != 0 )
        goto LABEL_12;
      goto LABEL_18;
    }
    v4 = a1[1];
  }
  v7 = a2[52];
  v8 = 56;
  v9 = a2[51];
  v19 = v7 != v9;
  v18 = **(_QWORD **)(v4 + 264);
  if ( (unsigned __int8)sub_1560180(v18 + 112, 56)
    || (v8 = 30, !(unsigned __int8)sub_1560180(v18 + 112, 30))
    || (*(_BYTE *)(v18 + 18) & 8) != 0 )
  {
    if ( v7 == v9 )
      goto LABEL_4;
    goto LABEL_12;
  }
LABEL_18:
  if ( !v19 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 96LL))(v6);
    goto LABEL_4;
  }
LABEL_12:
  if ( v3 )
  {
    v13 = sub_396EAF0(a1[1], v3);
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1[1] + 256) + 256LL))(
      *(_QWORD *)(a1[1] + 256),
      v13,
      8);
    v8 = v13;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 104LL))(v6, v13);
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 120LL))(v6);
  sub_39AB100(a1, v8, v14, v15, v16, v17);
LABEL_4:
  result = *(_QWORD *)(a1[1] + 240);
  if ( *(_DWORD *)(result + 348) == 3 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 88LL))(v6);
  return result;
}
