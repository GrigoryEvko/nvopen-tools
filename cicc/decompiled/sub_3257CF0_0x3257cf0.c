// Function: sub_3257CF0
// Address: 0x3257cf0
//
__int64 __fastcall sub_3257CF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rax
  const char *v4; // rax
  __int64 v5; // rdi
  const char *v6; // rdx
  const char *v8[4]; // [rsp-38h] [rbp-38h] BYREF
  __int16 v9; // [rsp-18h] [rbp-18h]
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    v6 = 0;
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL);
    v4 = 0;
    goto LABEL_6;
  }
  v3 = *(_QWORD *)(a2 - 8);
  if ( *(_QWORD *)v3 <= 5u || *(_DWORD *)(v3 + 24) != 1835622239 || *(_WORD *)(v3 + 28) != 24432 )
  {
    v4 = (const char *)(v3 + 24);
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL);
    v6 = (const char *)*((_QWORD *)v4 - 3);
LABEL_6:
    v10 = v2;
    v8[2] = v4;
    v8[0] = "__imp_";
    v8[3] = v6;
    v9 = 1283;
    return sub_E65280(v5, v8);
  }
  return 0;
}
