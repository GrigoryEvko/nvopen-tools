// Function: sub_1777F70
// Address: 0x1777f70
//
unsigned __int8 *__fastcall sub_1777F70(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 a5, __int64 *a6)
{
  unsigned __int8 *v9; // r12
  __int64 v10; // rax
  unsigned __int8 *v12; // rax
  __int64 v13; // rdi
  unsigned __int64 *v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rdx
  bool v18; // zf
  __int64 v19; // rsi
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  const void *v22; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v26; // [rsp+28h] [rbp-58h] BYREF
  char v27[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v28; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v28 = 257;
    v12 = (unsigned __int8 *)sub_1648A60(88, 2u);
    v9 = v12;
    if ( v12 )
    {
      v22 = a4;
      v24 = (__int64)v12;
      sub_15F1EA0((__int64)v12, *(_QWORD *)a2, 63, (__int64)(v12 - 48), 2, 0);
      *((_QWORD *)v9 + 7) = v9 + 72;
      *((_QWORD *)v9 + 8) = 0x400000000LL;
      sub_15FAD90((__int64)v9, a2, a3, v22, a5, (__int64)v27);
    }
    else
    {
      v24 = 0;
    }
    v13 = *(_QWORD *)(a1 + 8);
    if ( v13 )
    {
      v14 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v13 + 40, (__int64)v9);
      v15 = *((_QWORD *)v9 + 3);
      v16 = *v14;
      *((_QWORD *)v9 + 4) = v14;
      v16 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v9 + 3) = v16 | v15 & 7;
      *(_QWORD *)(v16 + 8) = v9 + 24;
      *v14 = *v14 & 7 | (unsigned __int64)(v9 + 24);
    }
    sub_164B780(v24, a6);
    v18 = *(_QWORD *)(a1 + 80) == 0;
    v26 = v9;
    if ( v18 )
      sub_4263D6(v24, a6, v17);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v26);
    v19 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v26 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v26, v19, 2);
      v20 = *((_QWORD *)v9 + 6);
      if ( v20 )
        sub_161E7C0((__int64)(v9 + 48), v20);
      v21 = v26;
      *((_QWORD *)v9 + 6) = v26;
      if ( v21 )
        sub_1623210((__int64)&v26, v21, (__int64)(v9 + 48));
    }
  }
  else
  {
    v9 = (unsigned __int8 *)sub_15A3A20((__int64 *)a2, (__int64 *)a3, a4, a5, 0);
    v10 = sub_14DBA30((__int64)v9, *(_QWORD *)(a1 + 96), 0);
    if ( v10 )
      return (unsigned __int8 *)v10;
  }
  return v9;
}
