// Function: sub_17A3CD0
// Address: 0x17a3cd0
//
unsigned __int8 *__fastcall sub_17A3CD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int8 *v11; // r12
  __int64 v12; // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // rdi
  unsigned __int64 *v16; // r13
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rdx
  bool v20; // zf
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  __int64 v24; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v25; // [rsp+18h] [rbp-58h] BYREF
  char v26[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v27; // [rsp+30h] [rbp-40h]

  v9 = sub_1643360(*(_QWORD **)(a1 + 24));
  v10 = sub_159C470(v9, a4, 0);
  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(v10 + 16) > 0x10u )
  {
    v24 = v10;
    v27 = 257;
    v14 = (unsigned __int8 *)sub_1648A60(56, 3u);
    v11 = v14;
    if ( v14 )
      sub_15FA480((__int64)v14, (__int64 *)a2, a3, v24, (__int64)v26, 0);
    v15 = *(_QWORD *)(a1 + 8);
    if ( v15 )
    {
      v16 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v15 + 40, (__int64)v11);
      v17 = *((_QWORD *)v11 + 3);
      v18 = *v16;
      *((_QWORD *)v11 + 4) = v16;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v11 + 3) = v18 | v17 & 7;
      *(_QWORD *)(v18 + 8) = v11 + 24;
      *v16 = *v16 & 7 | (unsigned __int64)(v11 + 24);
    }
    sub_164B780((__int64)v11, a5);
    v20 = *(_QWORD *)(a1 + 80) == 0;
    v25 = v11;
    if ( v20 )
      sub_4263D6(v11, a5, v19);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v25);
    v21 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v25 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v25, v21, 2);
      v22 = *((_QWORD *)v11 + 6);
      if ( v22 )
        sub_161E7C0((__int64)(v11 + 48), v22);
      v23 = v25;
      *((_QWORD *)v11 + 6) = v25;
      if ( v23 )
        sub_1623210((__int64)&v25, v23, (__int64)(v11 + 48));
    }
  }
  else
  {
    v11 = (unsigned __int8 *)sub_15A3890((__int64 *)a2, a3, v10, 0);
    v12 = sub_14DBA30((__int64)v11, *(_QWORD *)(a1 + 96), 0);
    if ( v12 )
      return (unsigned __int8 *)v12;
  }
  return v11;
}
