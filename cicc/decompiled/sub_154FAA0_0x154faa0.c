// Function: sub_154FAA0
// Address: 0x154faa0
//
_BYTE *__fastcall sub_154FAA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  void *v8; // rdx
  unsigned int v9; // ecx
  unsigned __int8 *v10; // rcx
  _BYTE *result; // rax
  __int64 v12; // [rsp+0h] [rbp-60h] BYREF
  char v13; // [rsp+8h] [rbp-58h]
  char *v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h]
  __int64 v17; // [rsp+28h] [rbp-38h]

  v8 = *(void **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v8 <= 0xBu )
  {
    sub_16E7EE0(a1, "!DILocation(", 12);
  }
  else
  {
    qmemcpy(v8, "!DILocation(", 12);
    *(_QWORD *)(a1 + 24) += 12LL;
  }
  v17 = a5;
  v9 = *(_DWORD *)(a2 + 4);
  v12 = a1;
  v14 = ", ";
  v13 = 1;
  v15 = a3;
  v16 = a4;
  sub_154ADE0((__int64)&v12, "line", 4u, v9, 0);
  sub_154ADE0((__int64)&v12, "column", 6u, *(unsigned __int16 *)(a2 + 2), 1);
  sub_154F950((__int64)&v12, "scope", 5u, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)), 0);
  v10 = 0;
  if ( *(_DWORD *)(a2 + 8) == 2 )
    v10 = *(unsigned __int8 **)(a2 - 8);
  sub_154F950((__int64)&v12, "inlinedAt", 9u, v10, 1);
  result = *(_BYTE **)(a1 + 24);
  if ( *(_BYTE **)(a1 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 24);
  return result;
}
