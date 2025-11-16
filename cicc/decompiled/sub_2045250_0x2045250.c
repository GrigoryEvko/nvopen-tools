// Function: sub_2045250
// Address: 0x2045250
//
__int64 __fastcall sub_2045250(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r15
  __int64 (*v6)(void); // rax
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rdx
  int v11; // esi
  int v12; // edi
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // [rsp-40h] [rbp-40h]

  result = *(unsigned __int8 *)(a2 + 240);
  if ( *(_BYTE *)(a1 + 240) != (_BYTE)result )
  {
    v4 = 0;
    v6 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a3 + 32) + 16LL) + 112LL);
    if ( v6 != sub_1D00B10 )
    {
      v15 = a3;
      v14 = v6();
      a3 = v15;
      v4 = v14;
    }
    v7 = *(_QWORD *)(a3 + 16);
    (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v7 + 1392LL))(
      v7,
      v4,
      *(_QWORD *)(a1 + 192),
      *(_QWORD *)(a1 + 200),
      *(unsigned __int8 *)(a1 + 240));
    v9 = v8;
    (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v7 + 1392LL))(
      v7,
      v4,
      *(_QWORD *)(a2 + 192),
      *(_QWORD *)(a2 + 200),
      *(unsigned __int8 *)(a2 + 240));
    v11 = *(unsigned __int8 *)(a1 + 240);
    v12 = v11 - 14;
    v13 = v11 - 2;
    LOBYTE(v13) = (unsigned __int8)(v11 - 2) <= 5u;
    LOBYTE(v12) = (unsigned __int8)(v11 - 14) <= 0x47u;
    result = v12 | v13;
    if ( ((unsigned __int8)(*(_BYTE *)(a2 + 240) - 14) <= 0x47u || (unsigned __int8)(*(_BYTE *)(a2 + 240) - 2) <= 5u) != (_BYTE)result
      || v9 != v10 )
    {
      sub_16BD130("Unsupported asm: input constraint with a matching output constraint of incompatible type!", 1u);
    }
    *(_BYTE *)(a2 + 240) = v11;
  }
  return result;
}
