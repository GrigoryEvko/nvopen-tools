// Function: sub_16009D0
// Address: 0x16009d0
//
__int64 __fastcall sub_16009D0(__int64 a1)
{
  unsigned int v1; // r9d
  __int64 v2; // r12
  __int64 v3; // r13
  char v4; // r15
  unsigned __int8 v5; // r14
  __int16 v6; // bx
  __int64 result; // rax
  unsigned int v8; // r8d
  int v9; // [rsp+8h] [rbp-38h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v1 = *(unsigned __int16 *)(a1 + 18);
  v2 = *(_QWORD *)(a1 - 48);
  v3 = *(_QWORD *)(a1 - 24);
  v4 = *(_BYTE *)(a1 + 56);
  v5 = *(_WORD *)(a1 + 18) & 1;
  v6 = (v1 >> 7) & 7;
  v9 = 1 << (v1 >> 1) >> 1;
  result = sub_1648A60(64, 2);
  if ( result )
  {
    v8 = v9;
    v10 = result;
    sub_15F9480(result, v2, v3, v5, v8, v6, v4, 0);
    return v10;
  }
  return result;
}
