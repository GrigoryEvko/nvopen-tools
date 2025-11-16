// Function: sub_154FDC0
// Address: 0x154fdc0
//
_BYTE *__fastcall sub_154FDC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  void *v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r8
  unsigned __int8 *v13; // rcx
  _BYTE *result; // rax
  __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  char v16; // [rsp+8h] [rbp-58h]
  char *v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h]
  __int64 v20; // [rsp+28h] [rbp-38h]

  v8 = *(void **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v8 <= 0xEu )
  {
    sub_16E7EE0(a1, "!DIDerivedType(", 15);
  }
  else
  {
    qmemcpy(v8, "!DIDerivedType(", 15);
    *(_QWORD *)(a1 + 24) += 15LL;
  }
  v20 = a5;
  v17 = ", ";
  v15 = a1;
  v16 = 1;
  v18 = a3;
  v19 = a4;
  sub_1549850(&v15, a2);
  v9 = *(unsigned int *)(a2 + 8);
  v10 = *(_QWORD *)(a2 + 8 * (2 - v9));
  if ( v10 )
  {
    v10 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v9)));
    v12 = v11;
  }
  else
  {
    v12 = 0;
  }
  sub_154AC80(&v15, "name", 4u, v10, v12, 1);
  sub_154F950((__int64)&v15, "scope", 5u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 1);
  v13 = (unsigned __int8 *)a2;
  if ( *(_BYTE *)a2 != 15 )
    v13 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  sub_154F950((__int64)&v15, "file", 4u, v13, 1);
  sub_154ADE0((__int64)&v15, "line", 4u, *(_DWORD *)(a2 + 24), 1);
  sub_154F950((__int64)&v15, "baseType", 8u, *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))), 0);
  sub_154B000((__int64)&v15, "size", 4u, *(_QWORD *)(a2 + 32), 1);
  sub_154ADE0((__int64)&v15, "align", 5u, *(_DWORD *)(a2 + 48), 1);
  sub_154B000((__int64)&v15, "offset", 6u, *(_QWORD *)(a2 + 40), 1);
  sub_154B2B0(&v15, "flags", 5u, *(_DWORD *)(a2 + 28));
  sub_154F950((__int64)&v15, "extraData", 9u, *(unsigned __int8 **)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8))), 1);
  if ( *(_BYTE *)(a2 + 56) )
    sub_154ADE0((__int64)&v15, "dwarfAddressSpace", 0x11u, *(_DWORD *)(a2 + 52), 0);
  result = *(_BYTE **)(a1 + 24);
  if ( *(_BYTE **)(a1 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 24);
  return result;
}
