// Function: sub_12AA0C0
// Address: 0x12aa0c0
//
__int64 __fastcall sub_12AA0C0(__int64 *a1, unsigned __int16 a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  int v10; // r9d
  _QWORD *v11; // r12
  _QWORD *v12; // rax
  __int64 v13; // rax
  int v14; // eax
  int v15; // r9d
  __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rsi
  int v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+10h] [rbp-70h]
  int v24; // [rsp+10h] [rbp-70h]
  int v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+28h] [rbp-58h] BYREF
  char v28[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v29; // [rsp+40h] [rbp-40h]

  if ( a3[16] <= 0x10u && *(_BYTE *)(a4 + 16) <= 0x10u )
    return sub_15A37B0(a2, a3, a4, 0);
  v25 = a4;
  v29 = 257;
  v9 = sub_1648A60(56, 2);
  v10 = v25;
  v11 = (_QWORD *)v9;
  if ( v9 )
  {
    v26 = v9;
    v12 = *(_QWORD **)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
    {
      v22 = v10;
      v23 = v12[4];
      v13 = sub_1643320(*v12);
      v14 = sub_16463B0(v13, v23);
      v15 = v22;
    }
    else
    {
      v24 = v10;
      v14 = sub_1643320(*v12);
      v15 = v24;
    }
    sub_15FEC10((_DWORD)v11, v14, 51, a2, (_DWORD)a3, v15, (__int64)v28, 0);
  }
  else
  {
    v26 = 0;
  }
  v16 = a1[1];
  if ( v16 )
  {
    v17 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v16 + 40, v11);
    v18 = v11[3];
    v19 = *v17;
    v11[4] = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    v11[3] = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v11 + 3;
    *v17 = *v17 & 7 | (unsigned __int64)(v11 + 3);
  }
  sub_164B780(v26, a5);
  v20 = *a1;
  if ( *a1 )
  {
    v27 = *a1;
    sub_1623A60(&v27, v20, 2);
    if ( v11[6] )
      sub_161E7C0(v11 + 6);
    v21 = v27;
    v11[6] = v27;
    if ( v21 )
      sub_1623210(&v27, v21, v11 + 6);
  }
  return (__int64)v11;
}
