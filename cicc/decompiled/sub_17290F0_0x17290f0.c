// Function: sub_17290F0
// Address: 0x17290f0
//
unsigned __int8 *__fastcall sub_17290F0(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  unsigned __int8 *v9; // r12
  __int64 v10; // rax
  unsigned __int8 *v12; // rax
  __int64 v13; // rdx
  _QWORD **v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // r8d
  __int64 v19; // rdi
  unsigned __int64 *v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rdx
  bool v24; // zf
  __int64 v25; // rsi
  __int64 v26; // rsi
  unsigned __int8 *v27; // rsi
  __int64 v28; // [rsp+0h] [rbp-80h]
  _QWORD *v29; // [rsp+8h] [rbp-78h]
  int v30; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v35; // [rsp+28h] [rbp-58h] BYREF
  char v36[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v37; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v37 = 257;
    v12 = (unsigned __int8 *)sub_1648A60(56, 2u);
    v13 = a4;
    v9 = v12;
    if ( v12 )
    {
      v33 = (__int64)v12;
      v14 = *(_QWORD ***)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      {
        v28 = v13;
        v29 = v14[4];
        v15 = (__int64 *)sub_1643320(*v14);
        v16 = (__int64)sub_16463B0(v15, (unsigned int)v29);
        v17 = v28;
      }
      else
      {
        v31 = v13;
        v16 = sub_1643320(*v14);
        v17 = v31;
      }
      sub_15FEC10((__int64)v9, v16, 52, a2, a3, v17, (__int64)v36, 0);
    }
    else
    {
      v33 = 0;
    }
    v18 = *(_DWORD *)(a1 + 40);
    if ( a6 || (a6 = *(_QWORD *)(a1 + 32)) != 0 )
    {
      v30 = *(_DWORD *)(a1 + 40);
      sub_1625C10((__int64)v9, 3, a6);
      v18 = v30;
    }
    sub_15F2440((__int64)v9, v18);
    v19 = *(_QWORD *)(a1 + 8);
    if ( v19 )
    {
      v20 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v19 + 40, (__int64)v9);
      v21 = *((_QWORD *)v9 + 3);
      v22 = *v20;
      *((_QWORD *)v9 + 4) = v20;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v9 + 3) = v22 | v21 & 7;
      *(_QWORD *)(v22 + 8) = v9 + 24;
      *v20 = *v20 & 7 | (unsigned __int64)(v9 + 24);
    }
    sub_164B780(v33, a5);
    v24 = *(_QWORD *)(a1 + 80) == 0;
    v35 = v9;
    if ( v24 )
      sub_4263D6(v33, a5, v23);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v35);
    v25 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v35 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v35, v25, 2);
      v26 = *((_QWORD *)v9 + 6);
      if ( v26 )
        sub_161E7C0((__int64)(v9 + 48), v26);
      v27 = v35;
      *((_QWORD *)v9 + 6) = v35;
      if ( v27 )
        sub_1623210((__int64)&v35, v27, (__int64)(v9 + 48));
    }
  }
  else
  {
    v9 = (unsigned __int8 *)sub_15A37B0(a2, (_QWORD *)a3, (_QWORD *)a4, 0);
    v10 = sub_14DBA30((__int64)v9, *(_QWORD *)(a1 + 96), 0);
    if ( v10 )
      return (unsigned __int8 *)v10;
  }
  return v9;
}
