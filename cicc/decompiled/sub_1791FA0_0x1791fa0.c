// Function: sub_1791FA0
// Address: 0x1791fa0
//
_QWORD *__fastcall sub_1791FA0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v7; // ax
  bool v8; // cc
  __int16 v9; // bx
  unsigned __int8 *v10; // r12
  __int64 v11; // rax
  unsigned __int8 *v13; // rax
  _QWORD **v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  bool v21; // zf
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned __int8 *v25; // rsi
  _QWORD *v26; // [rsp+0h] [rbp-B0h]
  unsigned __int64 *v27; // [rsp+0h] [rbp-B0h]
  __int64 v28; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v29; // [rsp+18h] [rbp-98h] BYREF
  __int64 v30[2]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v31; // [rsp+30h] [rbp-80h]
  __int64 v32; // [rsp+40h] [rbp-70h] BYREF
  __int16 v33; // [rsp+50h] [rbp-60h]
  char v34[16]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v35; // [rsp+70h] [rbp-40h]

  v7 = sub_14AEAE0(a2, 0);
  v8 = *(_BYTE *)(a3 + 16) <= 0x10u;
  v33 = 257;
  v9 = v7;
  v31 = 257;
  if ( v8 && *(_BYTE *)(a4 + 16) <= 0x10u )
  {
    v10 = (unsigned __int8 *)sub_15A37B0(v7, (_QWORD *)a3, (_QWORD *)a4, 0);
    v11 = sub_14DBA30((__int64)v10, *(_QWORD *)(a1 + 96), 0);
    if ( v11 )
      v10 = (unsigned __int8 *)v11;
  }
  else
  {
    v35 = 257;
    v13 = (unsigned __int8 *)sub_1648A60(56, 2u);
    v10 = v13;
    if ( v13 )
    {
      v28 = (__int64)v13;
      v14 = *(_QWORD ***)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      {
        v26 = v14[4];
        v15 = (__int64 *)sub_1643320(*v14);
        v16 = (__int64)sub_16463B0(v15, (unsigned int)v26);
      }
      else
      {
        v16 = sub_1643320(*v14);
      }
      sub_15FEC10((__int64)v10, v16, 51, v9, a3, a4, (__int64)v34, 0);
    }
    else
    {
      v28 = 0;
    }
    v17 = *(_QWORD *)(a1 + 8);
    if ( v17 )
    {
      v27 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v17 + 40, (__int64)v10);
      v18 = *v27;
      v19 = *((_QWORD *)v10 + 3) & 7LL;
      *((_QWORD *)v10 + 4) = v27;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v10 + 3) = v18 | v19;
      *(_QWORD *)(v18 + 8) = v10 + 24;
      *v27 = *v27 & 7 | (unsigned __int64)(v10 + 24);
    }
    sub_164B780(v28, v30);
    v21 = *(_QWORD *)(a1 + 80) == 0;
    v29 = v10;
    if ( v21 )
      sub_4263D6(v28, v30, v20);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v29);
    v22 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v29 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v29, v22, 2);
      v23 = *((_QWORD *)v10 + 6);
      v24 = (__int64)(v10 + 48);
      if ( v23 )
      {
        sub_161E7C0((__int64)(v10 + 48), v23);
        v24 = (__int64)(v10 + 48);
      }
      v25 = v29;
      *((_QWORD *)v10 + 6) = v29;
      if ( v25 )
        sub_1623210((__int64)&v29, v25, v24);
    }
  }
  return sub_1707C10(a1, (__int64)v10, a3, a4, &v32, 0);
}
