// Function: sub_17AF270
// Address: 0x17af270
//
unsigned __int8 *__fastcall sub_17AF270(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  unsigned __int8 *v8; // r12
  __int64 v9; // rax
  unsigned __int8 *v11; // rax
  __int64 v12; // rdi
  unsigned __int64 *v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  bool v17; // zf
  __int64 v18; // rsi
  __int64 v19; // rsi
  unsigned __int8 *v20; // rsi
  unsigned __int8 *v22; // [rsp+18h] [rbp-58h] BYREF
  char v23[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v24; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v24 = 257;
    v11 = (unsigned __int8 *)sub_1648A60(56, 3u);
    v8 = v11;
    if ( v11 )
      sub_15FA660((__int64)v11, (_QWORD *)a2, a3, (_QWORD *)a4, (__int64)v23, 0);
    v12 = *(_QWORD *)(a1 + 8);
    if ( v12 )
    {
      v13 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v12 + 40, (__int64)v8);
      v14 = *((_QWORD *)v8 + 3);
      v15 = *v13;
      *((_QWORD *)v8 + 4) = v13;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v8 + 3) = v15 | v14 & 7;
      *(_QWORD *)(v15 + 8) = v8 + 24;
      *v13 = *v13 & 7 | (unsigned __int64)(v8 + 24);
    }
    sub_164B780((__int64)v8, a5);
    v17 = *(_QWORD *)(a1 + 80) == 0;
    v22 = v8;
    if ( v17 )
      sub_4263D6(v8, a5, v16);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v22);
    v18 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v22 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v22, v18, 2);
      v19 = *((_QWORD *)v8 + 6);
      if ( v19 )
        sub_161E7C0((__int64)(v8 + 48), v19);
      v20 = v22;
      *((_QWORD *)v8 + 6) = v22;
      if ( v20 )
        sub_1623210((__int64)&v22, v20, (__int64)(v8 + 48));
    }
  }
  else
  {
    v8 = (unsigned __int8 *)sub_15A3950(a2, a3, (_BYTE *)a4, 0);
    v9 = sub_14DBA30((__int64)v8, *(_QWORD *)(a1 + 96), 0);
    if ( v9 )
      return (unsigned __int8 *)v9;
  }
  return v8;
}
