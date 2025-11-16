// Function: sub_190A170
// Address: 0x190a170
//
__int64 __fastcall sub_190A170(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned __int16 *v4; // r15
  unsigned __int16 *v6; // rdi
  __int64 v9; // rax
  int v10; // edi
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rdx
  unsigned int v14; // eax
  _QWORD *v15; // rax
  __int64 v16; // r13
  _QWORD *v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int16 *v23; // [rsp+8h] [rbp-78h]
  unsigned int v24; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v25; // [rsp+20h] [rbp-60h] BYREF
  __int64 v26; // [rsp+28h] [rbp-58h] BYREF
  _BYTE v27[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v28; // [rsp+40h] [rbp-40h]

  v3 = *a1;
  if ( *(_BYTE *)(*a1 + 8) != 13 )
    return 0;
  v4 = (unsigned __int16 *)*(a2 - 3);
  v6 = (unsigned __int16 *)*(a1 - 3);
  v25 = 0;
  v26 = 0;
  v23 = sub_14AC610(v6, &v25, a3);
  if ( v23 != sub_14AC610(v4, &v26, a3) )
    return 0;
  if ( v25 )
    return 0;
  if ( !v26 )
    return 0;
  v9 = sub_15A9930(a3, v3);
  v10 = *(_DWORD *)(v3 + 12);
  v24 = 0;
  v11 = *a2;
  v12 = v9;
  if ( !v10 )
    return 0;
  v13 = 0;
  v14 = 1;
  while ( v26 != *(_QWORD *)(v12 + v13 + 16) || v11 != *(_QWORD *)(*(_QWORD *)(v3 + 16) + v13) )
  {
    v24 = v14;
    v13 += 8;
    if ( v10 == v14 )
      return 0;
    ++v14;
  }
  v28 = 257;
  v15 = sub_1648A60(88, 1u);
  v16 = (__int64)v15;
  if ( v15 )
  {
    v17 = v15 - 3;
    v18 = sub_15FB2A0(*a1, &v24, 1);
    sub_15F1EA0(v16, v18, 62, v16 - 24, 1, (__int64)a2);
    if ( *(_QWORD *)(v16 - 24) )
    {
      v19 = *(_QWORD *)(v16 - 16);
      v20 = *(_QWORD *)(v16 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v20 = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
    }
    *(_QWORD *)(v16 - 24) = a1;
    v21 = a1[1];
    *(_QWORD *)(v16 - 16) = v21;
    if ( v21 )
      *(_QWORD *)(v21 + 16) = (v16 - 16) | *(_QWORD *)(v21 + 16) & 3LL;
    *(_QWORD *)(v16 - 8) = (unsigned __int64)(a1 + 1) | *(_QWORD *)(v16 - 8) & 3LL;
    a1[1] = (__int64)v17;
    *(_QWORD *)(v16 + 56) = v16 + 72;
    *(_QWORD *)(v16 + 64) = 0x400000000LL;
    sub_15FB110(v16, &v24, 1, (__int64)v27);
  }
  return v16;
}
