// Function: sub_14DA350
// Address: 0x14da350
//
__int64 __fastcall sub_14DA350(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  char *v11; // rdx
  char *v12; // r10
  char *v13; // r15
  __int64 *v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  char *v19; // [rsp+8h] [rbp-58h]
  unsigned __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  _QWORD v23[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = (a1 & 0xFFFFFFFFFFFFFFF8LL) + 56;
  v21 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a1 & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 21)
      || (v17 = *(_QWORD *)(v21 - 24), !*(_BYTE *)(v17 + 16))
      && (v23[0] = *(_QWORD *)(v17 + 112), (unsigned __int8)sub_1560260(v23, 0xFFFFFFFFLL, 21)) )
    {
      if ( !(unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 5) )
      {
        v18 = *(_QWORD *)(v21 - 24);
        if ( *(_BYTE *)(v18 + 16) )
          return 0;
        v23[0] = *(_QWORD *)(v18 + 112);
        if ( !(unsigned __int8)sub_1560260(v23, 0xFFFFFFFFLL, 5) )
          return 0;
      }
    }
    if ( (unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 52) )
      return 0;
    v10 = *(_QWORD *)(v21 - 24);
    if ( *(_BYTE *)(v10 + 16) )
      goto LABEL_12;
  }
  else
  {
    if ( (unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 21)
      || (v9 = *(_QWORD *)(v21 - 72), !*(_BYTE *)(v9 + 16))
      && (v23[0] = *(_QWORD *)(v9 + 112), (unsigned __int8)sub_1560260(v23, 0xFFFFFFFFLL, 21)) )
    {
      if ( !(unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 5) )
      {
        v16 = *(_QWORD *)(v21 - 72);
        if ( *(_BYTE *)(v16 + 16) )
          return 0;
        v23[0] = *(_QWORD *)(v16 + 112);
        if ( !(unsigned __int8)sub_1560260(v23, 0xFFFFFFFFLL, 5) )
          return 0;
      }
    }
    if ( (unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 52) )
      return 0;
    v10 = *(_QWORD *)(v21 - 72);
    if ( *(_BYTE *)(v10 + 16) )
      goto LABEL_12;
  }
  v23[0] = *(_QWORD *)(v10 + 112);
  if ( (unsigned __int8)sub_1560260(v23, 0xFFFFFFFFLL, 52) )
    return 0;
LABEL_12:
  if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 )
    return 0;
  v12 = (char *)sub_1649960(a2);
  v13 = v11;
  v14 = *(__int64 **)(*(_QWORD *)(a2 + 24) + 16LL);
  if ( *(_BYTE *)(*v14 + 8) != 16 )
    return sub_14D1BC0(v12, v11, *(unsigned int *)(a2 + 36), *v14, a3, a4, a5, a1);
  v19 = v12;
  v22 = *v14;
  v15 = (_BYTE *)sub_1632FA0(*(_QWORD *)(a2 + 40));
  return sub_14D8AA0(v19, v13, *(_DWORD *)(a2 + 36), v22, a3, a4, v15, a5, a1);
}
