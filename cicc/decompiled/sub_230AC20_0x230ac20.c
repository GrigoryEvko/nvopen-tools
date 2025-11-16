// Function: sub_230AC20
// Address: 0x230ac20
//
_QWORD *__fastcall sub_230AC20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 *v5; // rax
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rsi
  unsigned __int64 *i; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // r12
  unsigned __int64 v11; // r15
  _QWORD *v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  _QWORD *j; // rax
  unsigned __int64 *v16; // r15
  unsigned __int64 v17; // rcx
  unsigned __int64 *v18; // rax
  __int64 v19; // rcx
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rsi
  bool v22; // zf
  unsigned __int64 *v23; // r15
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 *v26; // rbx
  unsigned __int64 v27; // rdx
  _QWORD *v29; // [rsp+8h] [rbp-448h]
  unsigned __int64 *v30; // [rsp+8h] [rbp-448h]
  unsigned __int64 *v31; // [rsp+8h] [rbp-448h]
  _QWORD v32[5]; // [rsp+20h] [rbp-430h] BYREF
  char v33[8]; // [rsp+48h] [rbp-408h] BYREF
  unsigned __int64 v34; // [rsp+50h] [rbp-400h]
  char v35; // [rsp+64h] [rbp-3ECh]
  char v36[128]; // [rsp+68h] [rbp-3E8h] BYREF
  unsigned __int64 v37; // [rsp+E8h] [rbp-368h] BYREF
  unsigned __int64 *v38; // [rsp+F0h] [rbp-360h]
  char v39[8]; // [rsp+F8h] [rbp-358h] BYREF
  unsigned __int64 v40; // [rsp+100h] [rbp-350h]
  char v41; // [rsp+114h] [rbp-33Ch]
  char v42[264]; // [rsp+118h] [rbp-338h] BYREF
  __int64 v43; // [rsp+220h] [rbp-230h] BYREF
  __int64 v44; // [rsp+228h] [rbp-228h]
  __int64 v45; // [rsp+238h] [rbp-218h]
  __int64 v46; // [rsp+240h] [rbp-210h]
  char v47[8]; // [rsp+248h] [rbp-208h] BYREF
  unsigned __int64 v48; // [rsp+250h] [rbp-200h]
  char v49; // [rsp+264h] [rbp-1ECh]
  _BYTE v50[128]; // [rsp+268h] [rbp-1E8h] BYREF
  unsigned __int64 v51; // [rsp+2E8h] [rbp-168h] BYREF
  unsigned __int64 *v52; // [rsp+2F0h] [rbp-160h]
  char v53[8]; // [rsp+2F8h] [rbp-158h] BYREF
  unsigned __int64 v54; // [rsp+300h] [rbp-150h]
  char v55; // [rsp+314h] [rbp-13Ch]
  _BYTE v56[312]; // [rsp+318h] [rbp-138h] BYREF

  sub_22AC810((__int64)v32, a2 + 8, a3, a4, a5);
  v43 = v32[0];
  v44 = v32[1];
  v45 = v32[3];
  v46 = v32[4];
  sub_C8CF70((__int64)v47, v50, 16, (__int64)v36, (__int64)v33);
  v52 = &v51;
  v51 = (unsigned __int64)&v51 + 4;
  v5 = (__int64 *)v38;
  if ( v38 != &v37 )
  {
    v6 = v37 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*v38 & 0xFFFFFFFFFFFFFFF8LL) + 8) = &v37;
    v7 = v51;
    v37 = v37 & 7 | *v5 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v6 + 8) = &v51;
    v7 &= 0xFFFFFFFFFFFFFFF8LL;
    *v5 = v7 | *v5 & 7;
    *(_QWORD *)(v7 + 8) = v5;
    v51 = v6 | v51 & 7;
  }
  sub_C8CF70((__int64)v53, v56, 32, (__int64)v42, (__int64)v39);
  for ( i = v52; i != &v51; i = (unsigned __int64 *)i[1] )
  {
    if ( !i )
    {
      MEMORY[0x30] = &v43;
      BUG();
    }
    i[2] = (unsigned __int64)&v43;
  }
  v9 = (_QWORD *)sub_22077B0(0x200u);
  v10 = v9;
  if ( v9 )
  {
    v11 = (unsigned __int64)(v9 + 26);
    v29 = v9 + 1;
    *v9 = &unk_4A0AC78;
    v9[1] = v43;
    v9[2] = v44;
    v9[4] = v45;
    v9[5] = v46;
    sub_C8CF70((__int64)(v9 + 6), v9 + 10, 16, (__int64)v50, (__int64)v47);
    v10[27] = v11;
    v10[26] = v11 | 4;
    v12 = v52;
    if ( v52 != &v51 )
    {
      v13 = v51 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*v52 & 0xFFFFFFFFFFFFFFF8LL) + 8) = &v51;
      v14 = v10[26];
      v51 = v51 & 7 | *v12 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v13 + 8) = v11;
      v14 &= 0xFFFFFFFFFFFFFFF8LL;
      *v12 = v14 | *v12 & 7LL;
      *(_QWORD *)(v14 + 8) = v12;
      v10[26] = v13 | v10[26] & 7LL;
    }
    sub_C8CF70((__int64)(v10 + 28), v10 + 32, 32, (__int64)v56, (__int64)v53);
    for ( j = (_QWORD *)v10[27]; (_QWORD *)v11 != j; j = (_QWORD *)j[1] )
    {
      if ( !j )
      {
        MEMORY[0x30] = v29;
        BUG();
      }
      j[2] = v29;
    }
  }
  if ( !v55 )
    _libc_free(v54);
  v16 = v52;
  while ( v16 != &v51 )
  {
    v18 = v16;
    v16 = (unsigned __int64 *)v16[1];
    v20 = (unsigned __int64)(v18 - 4);
    v21 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
    *v16 = v21 | *v16 & 7;
    *(_QWORD *)(v21 + 8) = v16;
    *v18 &= 7u;
    v22 = *((_BYTE *)v18 + 76) == 0;
    v18[1] = 0;
    *(v18 - 4) = (unsigned __int64)&unk_4A09CC0;
    if ( v22 )
    {
      v31 = v18;
      _libc_free(v18[7]);
      v18 = v31;
    }
    v17 = v18[5];
    if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
    {
      v30 = v18;
      sub_BD60C0(v18 + 3);
      v18 = v30;
    }
    *(v18 - 4) = (unsigned __int64)&unk_49DB368;
    v19 = *(v18 - 1);
    if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
      sub_BD60C0(v18 - 3);
    j_j___libc_free_0(v20);
  }
  if ( !v49 )
    _libc_free(v48);
  v22 = v41 == 0;
  *a1 = v10;
  if ( v22 )
    _libc_free(v40);
  v23 = v38;
  while ( v23 != &v37 )
  {
    v26 = v23;
    v23 = (unsigned __int64 *)v23[1];
    v27 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
    *v23 = v27 | *v23 & 7;
    *(_QWORD *)(v27 + 8) = v23;
    *v26 &= 7u;
    v22 = *((_BYTE *)v26 + 76) == 0;
    v26[1] = 0;
    *(v26 - 4) = (unsigned __int64)&unk_4A09CC0;
    if ( v22 )
      _libc_free(v26[7]);
    v24 = v26[5];
    if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
      sub_BD60C0(v26 + 3);
    *(v26 - 4) = (unsigned __int64)&unk_49DB368;
    v25 = *(v26 - 1);
    if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
      sub_BD60C0(v26 - 3);
    j_j___libc_free_0((unsigned __int64)(v26 - 4));
  }
  if ( !v35 )
    _libc_free(v34);
  return a1;
}
