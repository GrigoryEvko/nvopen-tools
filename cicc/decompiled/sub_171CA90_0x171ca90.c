// Function: sub_171CA90
// Address: 0x171ca90
//
unsigned __int8 *__fastcall sub_171CA90(__int64 a1, __int64 a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int8 *v7; // r12
  __int64 v8; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int64 *v13; // r14
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  bool v17; // zf
  __int64 v18; // rsi
  __int64 v19; // rsi
  unsigned __int8 *v20; // rsi
  unsigned __int8 *v21; // [rsp+8h] [rbp-48h] BYREF
  char v22[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v23; // [rsp+20h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v23 = 257;
    v11 = sub_15FB630((__int64 *)a2, (__int64)v22, 0);
    v12 = *(_QWORD *)(a1 + 8);
    v7 = (unsigned __int8 *)v11;
    if ( v12 )
    {
      v13 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v12 + 40, v11);
      v14 = *((_QWORD *)v7 + 3);
      v15 = *v13;
      *((_QWORD *)v7 + 4) = v13;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v7 + 3) = v15 | v14 & 7;
      *(_QWORD *)(v15 + 8) = v7 + 24;
      *v13 = *v13 & 7 | (unsigned __int64)(v7 + 24);
    }
    sub_164B780((__int64)v7, a3);
    v17 = *(_QWORD *)(a1 + 80) == 0;
    v21 = v7;
    if ( v17 )
      sub_4263D6(v7, a3, v16);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v21);
    v18 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v21 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v21, v18, 2);
      v19 = *((_QWORD *)v7 + 6);
      if ( v19 )
        sub_161E7C0((__int64)(v7 + 48), v19);
      v20 = v21;
      *((_QWORD *)v7 + 6) = v21;
      if ( v20 )
        sub_1623210((__int64)&v21, v20, (__int64)(v7 + 48));
    }
  }
  else
  {
    v7 = (unsigned __int8 *)sub_15A2B00((__int64 *)a2, a4, a5, a6);
    v8 = sub_14DBA30((__int64)v7, *(_QWORD *)(a1 + 96), 0);
    if ( v8 )
      return (unsigned __int8 *)v8;
  }
  return v7;
}
