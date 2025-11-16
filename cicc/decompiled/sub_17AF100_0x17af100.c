// Function: sub_17AF100
// Address: 0x17af100
//
unsigned __int8 *__fastcall sub_17AF100(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned __int8 *v7; // r12
  __int64 v8; // rax
  unsigned __int8 *v10; // rax
  __int64 v11; // rdi
  unsigned __int64 *v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  bool v16; // zf
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  unsigned __int8 *v20; // [rsp+8h] [rbp-58h] BYREF
  char v21[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v22; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v22 = 257;
    v10 = (unsigned __int8 *)sub_1648A60(56, 2u);
    v7 = v10;
    if ( v10 )
      sub_15FA320((__int64)v10, (_QWORD *)a2, a3, (__int64)v21, 0);
    v11 = *(_QWORD *)(a1 + 8);
    if ( v11 )
    {
      v12 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v11 + 40, (__int64)v7);
      v13 = *((_QWORD *)v7 + 3);
      v14 = *v12;
      *((_QWORD *)v7 + 4) = v12;
      v14 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v7 + 3) = v14 | v13 & 7;
      *(_QWORD *)(v14 + 8) = v7 + 24;
      *v12 = *v12 & 7 | (unsigned __int64)(v7 + 24);
    }
    sub_164B780((__int64)v7, a4);
    v16 = *(_QWORD *)(a1 + 80) == 0;
    v20 = v7;
    if ( v16 )
      sub_4263D6(v7, a4, v15);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v20);
    v17 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v20 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v20, v17, 2);
      v18 = *((_QWORD *)v7 + 6);
      if ( v18 )
        sub_161E7C0((__int64)(v7 + 48), v18);
      v19 = v20;
      *((_QWORD *)v7 + 6) = v20;
      if ( v19 )
        sub_1623210((__int64)&v20, v19, (__int64)(v7 + 48));
    }
  }
  else
  {
    v7 = (unsigned __int8 *)sub_15A37D0((_BYTE *)a2, a3, 0);
    v8 = sub_14DBA30((__int64)v7, *(_QWORD *)(a1 + 96), 0);
    if ( v8 )
      return (unsigned __int8 *)v8;
  }
  return v7;
}
