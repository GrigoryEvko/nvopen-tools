// Function: sub_A5CD70
// Address: 0xa5cd70
//
_BYTE *__fastcall sub_A5CD70(__int64 a1, __int64 a2, __int64 a3)
{
  void *v4; // rdx
  unsigned int v5; // ecx
  __int64 v6; // r14
  unsigned __int8 v7; // al
  __int64 *v8; // rdx
  unsigned __int8 v9; // al
  __int64 v10; // rcx
  _BYTE *result; // rax
  __int64 v12; // r14
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  char v14; // [rsp+8h] [rbp-38h]
  char *v15; // [rsp+10h] [rbp-30h]
  __int64 v16; // [rsp+18h] [rbp-28h]

  v4 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v4 <= 0xBu )
  {
    sub_CB6200(a1, "!DILocation(", 12);
  }
  else
  {
    qmemcpy(v4, "!DILocation(", 12);
    *(_QWORD *)(a1 + 32) += 12LL;
  }
  v16 = a3;
  v5 = *(_DWORD *)(a2 + 4);
  v13 = a1;
  v6 = a2 - 16;
  v14 = 1;
  v15 = ", ";
  sub_A537C0((__int64)&v13, "line", 4u, v5, 0);
  sub_A537C0((__int64)&v13, "column", 6u, *(unsigned __int16 *)(a2 + 2), 1);
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(__int64 **)(a2 - 32);
  else
    v8 = (__int64 *)(v6 - 8LL * ((v7 >> 2) & 0xF));
  sub_A5CC00((__int64)&v13, "scope", 5u, *v8, 0);
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
  {
    v10 = 0;
    if ( *(_DWORD *)(a2 - 24) != 2 )
      goto LABEL_7;
    v12 = *(_QWORD *)(a2 - 32);
    goto LABEL_11;
  }
  v10 = 0;
  if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) == 2 )
  {
    v12 = v6 - 8LL * ((v9 >> 2) & 0xF);
LABEL_11:
    v10 = *(_QWORD *)(v12 + 8);
  }
LABEL_7:
  sub_A5CC00((__int64)&v13, "inlinedAt", 9u, v10, 1);
  sub_A53370((__int64)&v13, "isImplicitCode", 0xEu, *(char *)(a2 + 1) < 0, 0x100u);
  result = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == result )
    return (_BYTE *)sub_CB6200(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 32);
  return result;
}
