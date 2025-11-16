// Function: sub_26EF040
// Address: 0x26ef040
//
__int64 __fastcall sub_26EF040(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // r13
  bool v6; // al
  __int64 v7; // rbx
  __int64 v8; // r14
  bool v11; // [rsp+17h] [rbp-39h]
  __int64 v12; // [rsp+18h] [rbp-38h]

  v3 = a3 + 24;
  v4 = *(_QWORD *)(a3 + 32);
  v12 = a3 + 8;
  v11 = 0;
  if ( a3 + 24 == v4 )
  {
    v7 = *(_QWORD *)(a3 + 16);
    if ( v7 == v12 )
      goto LABEL_15;
    goto LABEL_7;
  }
  do
  {
    v5 = v4;
    v4 = *(_QWORD *)(v4 + 8);
    v6 = sub_B2FC80(v5 - 56);
    if ( v6 && !*(_QWORD *)(v5 - 40) )
    {
      v11 = v6;
      sub_B2E860((_QWORD *)(v5 - 56));
    }
  }
  while ( v3 != v4 );
  v7 = *(_QWORD *)(a3 + 16);
  while ( v7 != v12 )
  {
LABEL_7:
    v8 = v7;
    v7 = *(_QWORD *)(v7 + 8);
    if ( sub_B2FC80(v8 - 56) && !*(_QWORD *)(v8 - 40) )
      sub_B30290(v8 - 56);
  }
  if ( !v11 )
  {
LABEL_15:
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  memset((void *)a1, 0, 0x60u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
