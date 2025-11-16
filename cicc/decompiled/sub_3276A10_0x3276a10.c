// Function: sub_3276A10
// Address: 0x3276a10
//
__int64 __fastcall sub_3276A10(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r13
  int v6; // ecx
  unsigned __int16 *v7; // rax
  __int64 v8; // rsi
  unsigned __int16 v9; // r15
  __int64 v10; // r14
  bool v11; // zf
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r12
  int v16; // [rsp+Ch] [rbp-54h]
  __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  int v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  int v20; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(v4 + 8);
  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v17 = v8;
  if ( v8 )
  {
    v16 = v6;
    sub_B96E90((__int64)&v17, v8, 1);
    v6 = v16;
  }
  v11 = *(_DWORD *)(v5 + 24) == 51;
  v12 = *a1;
  v18 = *(_DWORD *)(a2 + 72);
  if ( v11 )
  {
    v19 = 0;
    v20 = 0;
    v14 = sub_33F17F0(v12, 51, &v19, v9, v10);
    if ( v19 )
      sub_B91220((__int64)&v19, v19);
  }
  else
  {
    v19 = v5;
    v20 = v6;
    v13 = sub_3402EA0(v12, 226, (unsigned int)&v17, v9, v10, 0, (__int64)&v19, 1);
    if ( !v13 )
      v13 = sub_32762F0(a2, (int)&v17, *a1);
    v14 = v13;
  }
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v14;
}
