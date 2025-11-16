// Function: sub_12AB550
// Address: 0x12ab550
//
__int64 __fastcall sub_12AB550(__int64 a1, _QWORD *a2, int a3, unsigned __int64 *a4)
{
  __int64 v4; // rdx
  unsigned int v7; // r12d
  __int64 v9; // rsi
  char *v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v25; // [rsp+30h] [rbp-60h]
  _BYTE v26[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v27; // [rsp+50h] [rbp-40h]

  v4 = (unsigned int)(a3 - 189);
  v7 = 4049;
  v9 = *(_QWORD *)(a4[9] + 16);
  if ( (unsigned int)v4 <= 0xD )
    v7 = dword_4280FE0[v4];
  v27 = 257;
  v10 = sub_128F980((__int64)a2, v9);
  v11 = (_QWORD *)a2[4];
  v24[0] = v10;
  v12 = sub_126A190(v11, v7, 0, 0);
  v13 = sub_1285290(a2 + 6, *(_QWORD *)(v12 + 24), v12, (int)v24, 1, (__int64)v26, 0);
  v14 = sub_127A030(a2[4] + 8LL, *a4, 0);
  v25 = 257;
  if ( v14 != *(_QWORD *)v13 )
  {
    if ( *(_BYTE *)(v13 + 16) > 0x10u )
    {
      v27 = 257;
      v16 = sub_15FDBD0(37, v13, v14, v26, 0);
      v17 = a2[7];
      v13 = v16;
      if ( v17 )
      {
        v18 = (__int64 *)a2[8];
        sub_157E9D0(v17 + 40, v16);
        v19 = *(_QWORD *)(v13 + 24);
        v20 = *v18;
        *(_QWORD *)(v13 + 32) = v18;
        v20 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v13 + 24) = v20 | v19 & 7;
        *(_QWORD *)(v20 + 8) = v13 + 24;
        *v18 = *v18 & 7 | (v13 + 24);
      }
      sub_164B780(v13, v24);
      v21 = a2[6];
      if ( v21 )
      {
        v23 = a2[6];
        sub_1623A60(&v23, v21, 2);
        if ( *(_QWORD *)(v13 + 48) )
          sub_161E7C0(v13 + 48);
        v22 = v23;
        *(_QWORD *)(v13 + 48) = v23;
        if ( v22 )
          sub_1623210(&v23, v22, v13 + 48);
      }
    }
    else
    {
      v13 = sub_15A46C0(37, v13, v14, 0);
    }
  }
  *(_QWORD *)a1 = v13;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
