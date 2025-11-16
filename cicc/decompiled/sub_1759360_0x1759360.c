// Function: sub_1759360
// Address: 0x1759360
//
unsigned __int8 *__fastcall sub_1759360(__int64 a1, __int64 a2, __int64 **a3, char a4, __int64 *a5)
{
  unsigned __int8 *v6; // r12
  __int64 v7; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 *v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  bool v16; // zf
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  unsigned __int8 *v20; // [rsp+8h] [rbp-48h] BYREF
  char v21[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v22; // [rsp+20h] [rbp-30h]

  if ( a3 == *(__int64 ***)a2 )
    return (unsigned __int8 *)a2;
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v22 = 257;
    v10 = sub_15FE0A0((_QWORD *)a2, (__int64)a3, a4, (__int64)v21, 0);
    v11 = *(_QWORD *)(a1 + 8);
    v6 = (unsigned __int8 *)v10;
    if ( v11 )
    {
      v12 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v11 + 40, v10);
      v13 = *((_QWORD *)v6 + 3);
      v14 = *v12;
      *((_QWORD *)v6 + 4) = v12;
      v14 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v6 + 3) = v14 | v13 & 7;
      *(_QWORD *)(v14 + 8) = v6 + 24;
      *v12 = *v12 & 7 | (unsigned __int64)(v6 + 24);
    }
    sub_164B780((__int64)v6, a5);
    v16 = *(_QWORD *)(a1 + 80) == 0;
    v20 = v6;
    if ( v16 )
      sub_4263D6(v6, a5, v15);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v20);
    v17 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v20 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v20, v17, 2);
      v18 = *((_QWORD *)v6 + 6);
      if ( v18 )
        sub_161E7C0((__int64)(v6 + 48), v18);
      v19 = v20;
      *((_QWORD *)v6 + 6) = v20;
      if ( v19 )
        sub_1623210((__int64)&v20, v19, (__int64)(v6 + 48));
    }
  }
  else
  {
    v6 = (unsigned __int8 *)sub_15A4750((__int64 ***)a2, a3, a4);
    v7 = sub_14DBA30((__int64)v6, *(_QWORD *)(a1 + 96), 0);
    if ( v7 )
      return (unsigned __int8 *)v7;
  }
  return v6;
}
