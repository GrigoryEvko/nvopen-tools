// Function: sub_11F98E0
// Address: 0x11f98e0
//
__int64 __fastcall sub_11F98E0(__int64 **a1, __int64 **a2, char a3, void **a4)
{
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 **v10; // rdi
  __int64 *v11; // rdx
  __m128i *v13[2]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v14; // [rsp+20h] [rbp-80h] BYREF
  __int64 **v15; // [rsp+30h] [rbp-70h] BYREF
  __int64 **v16; // [rsp+38h] [rbp-68h]
  char v17; // [rsp+40h] [rbp-60h]
  _BYTE v18[8]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v19; // [rsp+50h] [rbp-50h]
  __int64 v20; // [rsp+58h] [rbp-48h]
  __int64 v21; // [rsp+60h] [rbp-40h]
  __int64 v22; // [rsp+68h] [rbp-38h]

  v15 = a1;
  v16 = a2;
  v18[0] = a3;
  sub_C7D6A0(0, 0, 8);
  v19 = 1;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  sub_C7D6A0(0, 0, 8);
  v17 = 0;
  sub_CA0F50((__int64 *)v13, a4);
  sub_11F4060(&v15, v13);
  v5 = *v16;
  v6 = **v16;
  v7 = *(_QWORD *)(v6 + 80);
  v8 = v6 + 72;
  if ( v6 + 72 != v7 )
  {
    while ( 1 )
    {
      v9 = v7 - 24;
      if ( !v7 )
        v9 = 0;
      if ( (unsigned __int8)sub_11F7FC0((__int64)v18, v9, (__int64)v5) )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
          break;
      }
      else
      {
        sub_11F8520((__int64 *)&v15, v9);
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
          break;
      }
      v5 = *v16;
    }
  }
  v10 = v15;
  v11 = v15[4];
  if ( (unsigned __int64)((char *)v15[3] - (char *)v11) <= 1 )
  {
    sub_CB6200((__int64)v15, "}\n", 2u);
  }
  else
  {
    *(_WORD *)v11 = 2685;
    v10[4] = (__int64 *)((char *)v10[4] + 2);
  }
  if ( (__int64 *)v13[0] != &v14 )
    j_j___libc_free_0(v13[0], v14 + 1);
  return sub_C7D6A0(v20, 16LL * (unsigned int)v22, 8);
}
