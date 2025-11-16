// Function: sub_1B96E90
// Address: 0x1b96e90
//
__int64 __fastcall sub_1B96E90(__int64 a1, __int64 ***a2)
{
  unsigned int v4; // r14d
  unsigned int v5; // ebx
  __int64 v6; // rax
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 *v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r12
  char *v17; // [rsp+10h] [rbp-A0h] BYREF
  char v18; // [rsp+20h] [rbp-90h]
  char v19; // [rsp+21h] [rbp-8Fh]
  __int64 *v20; // [rsp+30h] [rbp-80h] BYREF
  __int64 v21; // [rsp+38h] [rbp-78h]
  _BYTE v22[112]; // [rsp+40h] [rbp-70h] BYREF

  v4 = *(_DWORD *)(a1 + 88);
  v20 = (__int64 *)v22;
  v21 = 0x800000000LL;
  if ( v4 )
  {
    v5 = 0;
    do
    {
      v6 = sub_1643350(*(_QWORD **)(a1 + 120));
      v9 = sub_159C470(v6, ~v5 + v4, 0);
      v10 = (unsigned int)v21;
      if ( (unsigned int)v21 >= HIDWORD(v21) )
      {
        sub_16CD150((__int64)&v20, v22, 0, 8, v7, v8);
        v10 = (unsigned int)v21;
      }
      ++v5;
      v20[v10] = v9;
      v4 = *(_DWORD *)(a1 + 88);
      v11 = (unsigned int)(v21 + 1);
      LODWORD(v21) = v21 + 1;
    }
    while ( v4 > v5 );
    v12 = v20;
  }
  else
  {
    v11 = 0;
    v12 = (__int64 *)v22;
  }
  v19 = 1;
  v17 = "reverse";
  v18 = 3;
  v13 = sub_15A01B0(v12, v11);
  v14 = sub_1599EF0(*a2);
  v15 = sub_14C50F0((__int64 *)(a1 + 96), (__int64)a2, v14, v13, (__int64)&v17);
  if ( v20 != (__int64 *)v22 )
    _libc_free((unsigned __int64)v20);
  return v15;
}
