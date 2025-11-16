// Function: sub_1550050
// Address: 0x1550050
//
_BYTE *__fastcall sub_1550050(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r8
  unsigned __int8 *v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r8
  _BYTE *result; // rax
  __int64 v18; // [rsp+0h] [rbp-60h] BYREF
  char v19; // [rsp+8h] [rbp-58h]
  char *v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  __int64 v23; // [rsp+28h] [rbp-38h]

  sub_1263B40(a1, "!DICompositeType(");
  v23 = a5;
  v18 = a1;
  v20 = ", ";
  v19 = 1;
  v21 = a3;
  v22 = a4;
  sub_1549850(&v18, a2);
  v8 = *(unsigned int *)(a2 + 8);
  v9 = *(_QWORD *)(a2 + 8 * (2 - v8));
  if ( v9 )
  {
    v9 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v8)));
    v11 = v10;
  }
  else
  {
    v11 = 0;
  }
  sub_154AC80(&v18, "name", 4u, v9, v11, 1);
  sub_154F950((__int64)&v18, "scope", 5u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 1);
  v12 = (unsigned __int8 *)a2;
  if ( *(_BYTE *)a2 != 15 )
    v12 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  sub_154F950((__int64)&v18, "file", 4u, v12, 1);
  sub_154ADE0((__int64)&v18, "line", 4u, *(_DWORD *)(a2 + 24), 1);
  sub_154F950((__int64)&v18, "baseType", 8u, *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))), 1);
  sub_154B000((__int64)&v18, "size", 4u, *(_QWORD *)(a2 + 32), 1);
  sub_154ADE0((__int64)&v18, "align", 5u, *(_DWORD *)(a2 + 48), 1);
  sub_154B000((__int64)&v18, "offset", 6u, *(_QWORD *)(a2 + 40), 1);
  sub_154B2B0(&v18, "flags", 5u, *(_DWORD *)(a2 + 28));
  sub_154F950((__int64)&v18, "elements", 8u, *(unsigned __int8 **)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8))), 1);
  sub_154B110(&v18, "runtimeLang", 0xBu, *(unsigned int *)(a2 + 52), (__int64 (__fastcall *)(_QWORD))sub_14E77F0);
  sub_154F950((__int64)&v18, "vtableHolder", 0xCu, *(unsigned __int8 **)(a2 + 8 * (5LL - *(unsigned int *)(a2 + 8))), 1);
  sub_154F950(
    (__int64)&v18,
    "templateParams",
    0xEu,
    *(unsigned __int8 **)(a2 + 8 * (6LL - *(unsigned int *)(a2 + 8))),
    1);
  v13 = *(unsigned int *)(a2 + 8);
  v14 = *(_QWORD *)(a2 + 8 * (7 - v13));
  if ( v14 )
  {
    v14 = sub_161E970(*(_QWORD *)(a2 + 8 * (7 - v13)));
    v16 = v15;
  }
  else
  {
    v16 = 0;
  }
  sub_154AC80(&v18, "identifier", 0xAu, v14, v16, 1);
  sub_154F950(
    (__int64)&v18,
    "discriminator",
    0xDu,
    *(unsigned __int8 **)(a2 + 8 * (8LL - *(unsigned int *)(a2 + 8))),
    1);
  result = *(_BYTE **)(a1 + 24);
  if ( *(_BYTE **)(a1 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 24);
  return result;
}
