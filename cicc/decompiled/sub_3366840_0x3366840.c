// Function: sub_3366840
// Address: 0x3366840
//
__int64 __fastcall sub_3366840(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rdx
  __int16 v11; // ax
  unsigned __int16 v12; // cx
  bool v13; // cl

  result = *(unsigned __int16 *)(a1 + 240);
  if ( *(_WORD *)(a2 + 240) != (_WORD)result )
  {
    v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a3 + 40) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 16LL));
    v6 = *(_QWORD *)(a3 + 16);
    v7 = v5;
    (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v6 + 2496LL))(
      v6,
      v5,
      *(_QWORD *)(a1 + 192),
      *(_QWORD *)(a1 + 200),
      *(unsigned __int16 *)(a1 + 240));
    v9 = v8;
    (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v6 + 2496LL))(
      v6,
      v7,
      *(_QWORD *)(a2 + 192),
      *(_QWORD *)(a2 + 200),
      *(unsigned __int16 *)(a2 + 240));
    v11 = *(_WORD *)(a2 + 240);
    v12 = v11 - 2;
    if ( (unsigned __int16)(*(_WORD *)(a1 + 240) - 2) <= 0xE2u )
    {
      if ( v12 <= 7u
        || (unsigned __int16)(v11 - 17) <= 0x6Cu
        || (unsigned __int16)(v11 - 176) <= 0x1Fu
        || (unsigned __int16)(v11 - 10) <= 6u
        || (unsigned __int16)(v11 - 126) <= 0x31u )
      {
LABEL_8:
        if ( v9 == v10 )
        {
          result = *(unsigned __int16 *)(a1 + 240);
          *(_WORD *)(a2 + 240) = result;
          return result;
        }
LABEL_17:
        sub_C64ED0("Unsupported asm: input constraint with a matching output constraint of incompatible type!", 1u);
      }
      v13 = 1;
    }
    else
    {
      if ( v12 <= 7u )
        goto LABEL_17;
      if ( (unsigned __int16)(v11 - 17) <= 0x6Cu )
        goto LABEL_17;
      if ( (unsigned __int16)(v11 - 176) <= 0x1Fu )
        goto LABEL_17;
      v13 = (unsigned __int16)(v11 - 10) <= 6u || (unsigned __int16)(v11 - 126) <= 0x31u;
      if ( v13 )
        goto LABEL_17;
    }
    if ( (unsigned __int16)(v11 - 208) <= 0x14u != v13 )
      goto LABEL_17;
    goto LABEL_8;
  }
  return result;
}
