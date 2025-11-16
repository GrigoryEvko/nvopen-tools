// Function: sub_1AEC650
// Address: 0x1aec650
//
bool __fastcall sub_1AEC650(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // r12
  _QWORD *v3; // r14
  bool result; // al
  __int64 v5; // rax
  __int64 v6; // r15
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // [rsp+8h] [rbp-48h]
  unsigned int v13[14]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = (_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( ((a1 >> 2) & 1) != 0 )
  {
    result = sub_15602A0(v3, -1, "gc-leaf-function", 0x10u);
    if ( result )
      return result;
    v5 = *(_QWORD *)(v2 - 24);
    if ( *(_BYTE *)(v5 + 16) )
    {
      v12 = (__int64 *)(v2 - 24);
      goto LABEL_9;
    }
    *(_QWORD *)v13 = *(_QWORD *)(v5 + 112);
    if ( !sub_15602A0(v13, -1, "gc-leaf-function", 0x10u) )
    {
      v6 = *(_QWORD *)(v2 - 24);
      v12 = (__int64 *)(v2 - 24);
      if ( *(_BYTE *)(v6 + 16) )
        goto LABEL_9;
      goto LABEL_6;
    }
    return 1;
  }
  result = sub_15602A0(v3, -1, "gc-leaf-function", 0x10u);
  if ( result )
    return result;
  v9 = *(_QWORD *)(v2 - 72);
  if ( *(_BYTE *)(v9 + 16) )
  {
    v12 = (__int64 *)(v2 - 72);
    goto LABEL_20;
  }
  *(_QWORD *)v13 = *(_QWORD *)(v9 + 112);
  if ( sub_15602A0(v13, -1, "gc-leaf-function", 0x10u) )
    return 1;
  v6 = *(_QWORD *)(v2 - 72);
  v12 = (__int64 *)(v2 - 72);
  if ( *(_BYTE *)(v6 + 16) )
    goto LABEL_20;
LABEL_6:
  if ( sub_15602E0((_QWORD *)(v6 + 112), "gc-leaf-function", 0x10u) )
    return 1;
  v7 = *(_DWORD *)(v6 + 36);
  if ( v7 )
    return v7 != 75 && v7 != 78;
  if ( ((a1 >> 2) & 1) != 0 )
  {
LABEL_9:
    if ( (unsigned __int8)sub_1560260(v3, -1, 21)
      || (v8 = *(_QWORD *)(v2 - 24), !*(_BYTE *)(v8 + 16))
      && (*(_QWORD *)v13 = *(_QWORD *)(v8 + 112), (unsigned __int8)sub_1560260(v13, -1, 21)) )
    {
      if ( !(unsigned __int8)sub_1560260(v3, -1, 5) )
      {
        v11 = *(_QWORD *)(v2 - 24);
        if ( *(_BYTE *)(v11 + 16) )
          return 0;
LABEL_25:
        *(_QWORD *)v13 = *(_QWORD *)(v11 + 112);
        if ( !(unsigned __int8)sub_1560260(v13, -1, 5) )
          return 0;
        goto LABEL_12;
      }
    }
    goto LABEL_12;
  }
LABEL_20:
  if ( (unsigned __int8)sub_1560260(v3, -1, 21)
    || (v10 = *(_QWORD *)(v2 - 72), !*(_BYTE *)(v10 + 16))
    && (*(_QWORD *)v13 = *(_QWORD *)(v10 + 112), (unsigned __int8)sub_1560260(v13, -1, 21)) )
  {
    if ( !(unsigned __int8)sub_1560260(v3, -1, 5) )
    {
      v11 = *(_QWORD *)(v2 - 72);
      if ( *(_BYTE *)(v11 + 16) )
        return 0;
      goto LABEL_25;
    }
  }
LABEL_12:
  if ( *(_BYTE *)(*v12 + 16) || !sub_149CB50(*a2, *v12, v13) )
    return 0;
  return (((int)*(unsigned __int8 *)(*a2 + (signed int)v13[0] / 4) >> (2 * (v13[0] & 3))) & 3) != 0;
}
