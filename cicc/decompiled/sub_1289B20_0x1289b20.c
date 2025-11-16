// Function: sub_1289B20
// Address: 0x1289b20
//
__int64 __fastcall sub_1289B20(__int64 *a1, unsigned __int16 a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rax
  int v11; // edx
  _QWORD *v12; // r12
  _QWORD *v13; // rax
  __int64 v14; // rax
  int v15; // eax
  int v16; // edx
  unsigned int v17; // r8d
  __int64 v18; // rdi
  unsigned __int64 *v19; // r13
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rsi
  int v24; // [rsp+0h] [rbp-80h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  unsigned int v26; // [rsp+8h] [rbp-78h]
  int v27; // [rsp+8h] [rbp-78h]
  int v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+28h] [rbp-58h] BYREF
  char v32[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v33; // [rsp+40h] [rbp-40h]

  if ( a3[16] <= 0x10u && *(_BYTE *)(a4 + 16) <= 0x10u )
    return sub_15A37B0(a2, a3, a4, 0);
  v28 = a4;
  v33 = 257;
  v10 = sub_1648A60(56, 2);
  v11 = v28;
  v12 = (_QWORD *)v10;
  if ( v10 )
  {
    v29 = v10;
    v13 = *(_QWORD **)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
    {
      v24 = v11;
      v25 = v13[4];
      v14 = sub_1643320(*v13);
      v15 = sub_16463B0(v14, v25);
      v16 = v24;
    }
    else
    {
      v27 = v11;
      v15 = sub_1643320(*v13);
      v16 = v27;
    }
    sub_15FEC10((_DWORD)v12, v15, 52, a2, (_DWORD)a3, v16, (__int64)v32, 0);
  }
  else
  {
    v29 = 0;
  }
  v17 = *((_DWORD *)a1 + 10);
  if ( a6 || (a6 = a1[4]) != 0 )
  {
    v26 = *((_DWORD *)a1 + 10);
    sub_1625C10(v12, 3, a6);
    v17 = v26;
  }
  sub_15F2440(v12, v17);
  v18 = a1[1];
  if ( v18 )
  {
    v19 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v18 + 40, v12);
    v20 = v12[3];
    v21 = *v19;
    v12[4] = v19;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    v12[3] = v21 | v20 & 7;
    *(_QWORD *)(v21 + 8) = v12 + 3;
    *v19 = *v19 & 7 | (unsigned __int64)(v12 + 3);
  }
  sub_164B780(v29, a5);
  v22 = *a1;
  if ( *a1 )
  {
    v31 = *a1;
    sub_1623A60(&v31, v22, 2);
    if ( v12[6] )
      sub_161E7C0(v12 + 6);
    v23 = v31;
    v12[6] = v31;
    if ( v23 )
      sub_1623210(&v31, v23, v12 + 6);
  }
  return (__int64)v12;
}
