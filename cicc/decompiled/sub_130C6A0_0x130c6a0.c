// Function: sub_130C6A0
// Address: 0x130c6a0
//
__int64 __fastcall sub_130C6A0(__int64 a1, __int64 a2, __int64 a3, volatile signed __int64 *a4, __int64 a5, int a6)
{
  __int64 v9; // rax
  __int64 v10; // r12
  unsigned __int64 v11; // r12
  __int64 result; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // r12
  __int64 v15; // rax
  bool v16; // zf
  unsigned __int64 v17; // r12
  unsigned __int8 v19; // [rsp+18h] [rbp-48h]
  __int64 v20[7]; // [rsp+28h] [rbp-38h] BYREF

  v9 = *(_QWORD *)(a3 + 120);
  if ( v9 <= 0 )
  {
    if ( v9 )
      return 0;
    v14 = sub_13427E0(a5 + 112);
    v15 = sub_13427E0(a5 + 9768);
    v16 = v15 + v14 == 0;
    v17 = v15 + v14;
    if ( *(_BYTE *)(a3 + 112) | v16 )
      return 0;
    v19 = 0;
    sub_130C320(a1, a2, a3, a4, a5, 0, 0, v17);
    return v19;
  }
  sub_130B270(v20);
  v10 = sub_13427E0(a5 + 112);
  v11 = sub_13427E0(a5 + 9768) + v10;
  result = sub_133D9F0(a3, v20, v11);
  if ( !a6 || a6 == 2 && (_BYTE)result )
  {
    v13 = *(_QWORD *)(a3 + 160);
    if ( v13 < v11 && !*(_BYTE *)(a3 + 112) )
    {
      v19 = result;
      sub_130C320(a1, a2, a3, a4, a5, 0, v13, v11 - v13);
      return v19;
    }
  }
  return result;
}
