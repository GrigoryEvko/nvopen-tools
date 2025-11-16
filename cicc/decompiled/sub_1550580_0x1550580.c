// Function: sub_1550580
// Address: 0x1550580
//
_BYTE *__fastcall sub_1550580(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  void *v8; // rdx
  bool v9; // zf
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _BYTE *result; // rax
  __int64 v18; // [rsp+0h] [rbp-60h] BYREF
  __int64 v19; // [rsp+8h] [rbp-58h]
  char *v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  __int64 v23; // [rsp+28h] [rbp-38h]

  v8 = *(void **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v8 <= 0xDu )
  {
    sub_16E7EE0(a1, "!DIStringType(", 14);
  }
  else
  {
    qmemcpy(v8, "!DIStringType(", 14);
    *(_QWORD *)(a1 + 24) += 14LL;
  }
  v9 = *(_WORD *)(a2 + 2) == 18;
  v23 = a5;
  v18 = a1;
  LOBYTE(v19) = 1;
  v20 = ", ";
  v21 = a3;
  v22 = a4;
  if ( !v9 )
    sub_1549850(&v18, a2);
  v10 = *(unsigned int *)(a2 + 8);
  v11 = *(_QWORD *)(a2 + 8 * (2 - v10));
  if ( v11 )
  {
    v11 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v10)));
    v13 = v12;
  }
  else
  {
    v13 = 0;
  }
  sub_154AC80(&v18, "name", 4u, v11, v13, 1);
  sub_154F950((__int64)&v18, "stringLength", 0xCu, *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))), 1);
  sub_154F950(
    (__int64)&v18,
    "stringLengthExpression",
    0x16u,
    *(unsigned __int8 **)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8))),
    1);
  sub_154B000((__int64)&v18, "size", 4u, *(_QWORD *)(a2 + 32), 1);
  sub_154ADE0((__int64)&v18, "align", 5u, *(_DWORD *)(a2 + 48), 1);
  sub_154B110(&v18, "encoding", 8u, *(unsigned int *)(a2 + 52), (__int64 (__fastcall *)(_QWORD))sub_14E6F20);
  result = *(_BYTE **)(a1 + 24);
  if ( *(_BYTE **)(a1 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a1, ")", 1, v14, v15, v16, v18, v19, v20, v21, v22, v23);
  *result = 41;
  ++*(_QWORD *)(a1 + 24);
  return result;
}
