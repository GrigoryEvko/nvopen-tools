// Function: sub_2574D70
// Address: 0x2574d70
//
__int64 __fastcall sub_2574D70(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  char v13; // [rsp+7h] [rbp-29h] BYREF
  __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_25096F0((_QWORD *)(a1 + 72));
  if ( (*(_BYTE *)(sub_25096F0((_QWORD *)(a1 + 72)) + 32) & 0xFu) - 7 > 1
    || (v13 = 0,
        result = sub_2523890(a2, (__int64 (__fastcall *)(__int64, __int64 *))sub_253A4F0, (__int64)v14, a1, 1u, &v13),
        !(_BYTE)result) )
  {
    v8 = *(_QWORD *)(v3 + 80);
    v9 = a1 + 104;
    if ( !v8 )
      BUG();
    v10 = *(_QWORD *)(v8 + 32);
    v11 = v10 - 24;
    if ( v10 )
      v10 -= 24;
    v14[0] = v10;
    sub_2574950(v9, v14, v11, v4, v5, v6);
    v12 = *(_QWORD *)(v3 + 80);
    if ( v12 )
      v12 -= 24;
    return sub_256DB10(a1, a2, v12);
  }
  return result;
}
