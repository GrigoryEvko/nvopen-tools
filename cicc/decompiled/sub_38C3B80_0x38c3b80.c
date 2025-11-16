// Function: sub_38C3B80
// Address: 0x38c3b80
//
__int64 __fastcall sub_38C3B80(__int64 a1, __int64 a2, int a3, int a4, int a5, __int64 a6, int a7, __int64 a8)
{
  __int64 v9; // rax
  _BYTE *v10; // r9
  __int64 v11; // rax
  int v13; // [rsp+Ch] [rbp-64h]
  int v14; // [rsp+10h] [rbp-60h]
  int v15; // [rsp+10h] [rbp-60h]
  int v16; // [rsp+10h] [rbp-60h]
  int v17; // [rsp+14h] [rbp-5Ch]
  int v18; // [rsp+14h] [rbp-5Ch]
  int v19; // [rsp+14h] [rbp-5Ch]
  int v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+18h] [rbp-58h]
  int v22; // [rsp+18h] [rbp-58h]
  __int64 *v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a6 + 16) <= 1u )
  {
    v10 = 0;
  }
  else
  {
    v14 = a5;
    v17 = a4;
    v20 = a3;
    sub_16E2FC0((__int64 *)&v23, a6);
    a3 = v20;
    v9 = v24;
    a4 = v17;
    a5 = v14;
    if ( v23 != &v25 )
    {
      v21 = v24;
      v13 = v14;
      v15 = v17;
      v18 = a3;
      j_j___libc_free_0((unsigned __int64)v23);
      a5 = v13;
      a4 = v15;
      a3 = v18;
      v9 = v21;
    }
    v10 = 0;
    if ( v9 )
    {
      v16 = a5;
      v19 = a4;
      v22 = a3;
      v11 = sub_38BF510(a1, a6);
      a5 = v16;
      a4 = v19;
      a3 = v22;
      v10 = (_BYTE *)v11;
    }
  }
  return sub_38C37C0(a1, a2, a3, a4, a5, v10, a7, a8);
}
