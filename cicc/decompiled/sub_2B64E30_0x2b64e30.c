// Function: sub_2B64E30
// Address: 0x2b64e30
//
__int64 __fastcall sub_2B64E30(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v5; // r13d
  int v6; // r15d
  int v7; // eax
  unsigned __int8 *v8; // rdi
  unsigned __int8 *v9; // r10
  unsigned __int8 *v10; // r12
  unsigned __int8 *v11; // rbx
  __int64 result; // rax
  int v13; // [rsp+Ch] [rbp-44h]
  unsigned __int8 *v14; // [rsp+10h] [rbp-40h]

  v5 = *(_WORD *)(a2 + 2) & 0x3F;
  v6 = *(_WORD *)(a1 + 2) & 0x3F;
  v7 = sub_B52F50(v5);
  v8 = *(unsigned __int8 **)(a1 - 64);
  v9 = *(unsigned __int8 **)(a2 - 64);
  v10 = *(unsigned __int8 **)(a1 - 32);
  v11 = *(unsigned __int8 **)(a2 - 32);
  if ( v5 != v6 )
  {
    if ( v7 != v6 )
      return 0;
    return sub_2B64D40(v8, v10, v11, v9, a3);
  }
  v13 = v7;
  v14 = *(unsigned __int8 **)(a2 - 64);
  result = sub_2B64D40(v8, v10, v9, v11, a3);
  if ( !(_BYTE)result )
  {
    v9 = v14;
    if ( v13 != v6 )
      return 0;
    return sub_2B64D40(v8, v10, v11, v9, a3);
  }
  return result;
}
