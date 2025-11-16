// Function: sub_27DBB30
// Address: 0x27dbb30
//
bool __fastcall sub_27DBB30(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r13
  __int64 v5; // rdi
  bool v6; // r14
  bool result; // al
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rcx
  _BYTE *v11; // rdi

  v3 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  v4 = *(_QWORD *)(a1 - 96);
  if ( !sub_BCAC40(v3, 1) )
  {
LABEL_12:
    v5 = *(_QWORD *)(a1 + 8);
    goto LABEL_13;
  }
  if ( *(_BYTE *)a1 == 57 )
  {
LABEL_9:
    v6 = 1;
    goto LABEL_10;
  }
  v5 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)a1 == 86 && *(_QWORD *)(*(_QWORD *)(a1 - 96) + 8LL) == v5 && **(_BYTE **)(a1 - 32) <= 0x15u )
  {
    if ( sub_AC30F0(*(_QWORD *)(a1 - 32)) )
      goto LABEL_9;
    goto LABEL_12;
  }
LABEL_13:
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  v6 = sub_BCAC40(v5, 1);
  if ( v6 )
  {
    if ( *(_BYTE *)a1 == 58 )
      goto LABEL_9;
    v6 = 0;
    if ( *(_BYTE *)a1 == 86 )
    {
      v10 = *(_QWORD *)(a1 + 8);
      if ( *(_QWORD *)(*(_QWORD *)(a1 - 96) + 8LL) == v10 )
      {
        v11 = *(_BYTE **)(a1 - 64);
        if ( *v11 <= 0x15u )
          v6 = sub_AD7A80(v11, 1, v8, v10, v9);
      }
    }
  }
LABEL_10:
  result = v4 == a2 && v4 != 0;
  if ( result )
    return !v6 & sub_BCAC40(*(_QWORD *)(v4 + 8), 1);
  return result;
}
