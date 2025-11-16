// Function: sub_38C2CC0
// Address: 0x38c2cc0
//
__int64 __fastcall sub_38C2CC0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, int a5)
{
  __int64 v8; // rax
  _BYTE *v9; // rcx
  __int64 v10; // rax
  __int64 v12; // [rsp+8h] [rbp-58h]
  __int64 *v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a4 + 16) <= 1u )
  {
    v9 = 0;
  }
  else
  {
    sub_16E2FC0((__int64 *)&v13, a4);
    v8 = v14;
    if ( v13 != &v15 )
    {
      v12 = v14;
      j_j___libc_free_0((unsigned __int64)v13);
      v8 = v12;
    }
    v9 = 0;
    if ( v8 )
    {
      v10 = sub_38BF510(a1, a4);
      *(_BYTE *)(v10 + 38) = 1;
      v9 = (_BYTE *)v10;
    }
  }
  return sub_38C2930(a1, a2, a3, v9, a5);
}
