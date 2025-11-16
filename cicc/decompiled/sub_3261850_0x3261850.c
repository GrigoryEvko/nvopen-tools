// Function: sub_3261850
// Address: 0x3261850
//
__int64 __fastcall sub_3261850(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rcx
  int v5; // r8d
  unsigned __int16 *v6; // rax
  __int64 v7; // rsi
  int v8; // r14d
  __int64 v9; // r15
  int v10; // eax
  __int64 v11; // rdi
  int v12; // ecx
  __int64 v13; // r14
  __int64 v14; // rax
  int v16; // [rsp+4h] [rbp-5Ch]
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  int v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  int v21; // [rsp+28h] [rbp-38h]

  v3 = *(__int64 **)(a2 + 40);
  v4 = *v3;
  v5 = *((_DWORD *)v3 + 2);
  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v18 = v7;
  if ( v7 )
  {
    v16 = v5;
    v17 = v4;
    sub_B96E90((__int64)&v18, v7, 1);
    v5 = v16;
    v4 = v17;
  }
  v10 = *(_DWORD *)(a2 + 72);
  v11 = *a1;
  v20 = v4;
  v12 = v8;
  v19 = v10;
  v13 = 0;
  v21 = v5;
  v14 = sub_3402EA0(v11, 204, (unsigned int)&v18, v12, v9, 0, (__int64)&v20, 1);
  if ( v14 )
    v13 = v14;
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v13;
}
