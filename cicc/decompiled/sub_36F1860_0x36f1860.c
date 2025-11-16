// Function: sub_36F1860
// Address: 0x36f1860
//
__int64 __fastcall sub_36F1860(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  unsigned __int8 v4; // r15
  __int64 *v5; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // r13
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // [rsp+10h] [rbp-70h]
  const char *v17; // [rsp+20h] [rbp-60h] BYREF
  __int64 v18; // [rsp+28h] [rbp-58h]
  __int16 v19; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)a1 == 22 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 80LL);
    if ( !v2 )
      BUG();
    v3 = *(_QWORD *)(v2 + 32);
    v4 = 1;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 32);
    v4 = 0;
  }
  v5 = (__int64 *)sub_BD5C60(a1);
  v6 = sub_BCE3C0(v5, a2);
  v17 = sub_BD5D20(a1);
  v18 = v7;
  v19 = 261;
  v8 = sub_BD2C40(72, 1u);
  v9 = (__int64)v8;
  if ( v8 )
    sub_B51C90((__int64)v8, a1, v6, (__int64)&v17, v3, v4);
  v16 = *(_QWORD *)(a1 + 8);
  v17 = sub_BD5D20(a1);
  v18 = v10;
  v19 = 261;
  v11 = sub_BD2C40(72, 1u);
  v12 = (__int64)v11;
  if ( v11 )
    sub_B51C90((__int64)v11, v9, v16, (__int64)&v17, v3, v4);
  sub_BD84D0(a1, v12);
  if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
    result = *(_QWORD *)(v9 - 8);
  else
    result = v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)result )
  {
    v14 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = a1;
  v15 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(result + 8) = v15;
  if ( v15 )
    *(_QWORD *)(v15 + 16) = result + 8;
  *(_QWORD *)(result + 16) = a1 + 16;
  *(_QWORD *)(a1 + 16) = result;
  return result;
}
