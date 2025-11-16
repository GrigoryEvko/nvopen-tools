// Function: sub_2F920F0
// Address: 0x2f920f0
//
__int64 __fastcall sub_2F920F0(__int64 a1, __int64 *a2, __int64 *a3, int a4)
{
  __int64 v7; // r8
  __int64 *v8; // rsi
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // r9
  __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  int v15; // [rsp+8h] [rbp-28h]
  int v16; // [rsp+Ch] [rbp-24h]

  v7 = *a2;
  v8 = (__int64 *)(a1 + 2216);
  v9 = *a3;
  if ( !*(_BYTE *)(a1 + 2896) )
    v8 = 0;
  result = sub_2E8A880(v7, v8, v9, qword_50254C8);
  if ( (_BYTE)result )
  {
    v14 = (unsigned __int64)a2 | 6;
    v16 = a4;
    v15 = 1;
    return sub_2F8F1B0((__int64)a3, (__int64)&v14, 1u, v11, v12, v13);
  }
  return result;
}
