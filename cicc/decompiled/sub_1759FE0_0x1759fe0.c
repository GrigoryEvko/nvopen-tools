// Function: sub_1759FE0
// Address: 0x1759fe0
//
unsigned __int8 *__fastcall sub_1759FE0(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 *a5)
{
  unsigned __int8 *v7; // r12
  __int64 v8; // rax
  unsigned __int8 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 *v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  bool v22; // zf
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v29; // [rsp+28h] [rbp-58h] BYREF
  char v30[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v31; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v31 = 257;
    v11 = (unsigned __int8 *)sub_1648A60(88, 1u);
    v7 = v11;
    if ( v11 )
    {
      v12 = a4;
      v26 = a4;
      v28 = (__int64)v11;
      v13 = sub_15FB2A0(*(_QWORD *)a2, a3, v12);
      sub_15F1EA0((__int64)v7, v13, 62, (__int64)(v7 - 24), 1, 0);
      if ( *((_QWORD *)v7 - 3) )
      {
        v14 = *((_QWORD *)v7 - 2);
        v15 = *((_QWORD *)v7 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v15 = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
      }
      *((_QWORD *)v7 - 3) = a2;
      v16 = *(_QWORD *)(a2 + 8);
      *((_QWORD *)v7 - 2) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = (unsigned __int64)(v7 - 16) | *(_QWORD *)(v16 + 16) & 3LL;
      *((_QWORD *)v7 - 1) = (a2 + 8) | *((_QWORD *)v7 - 1) & 3LL;
      *(_QWORD *)(a2 + 8) = v7 - 24;
      *((_QWORD *)v7 + 7) = v7 + 72;
      *((_QWORD *)v7 + 8) = 0x400000000LL;
      sub_15FB110((__int64)v7, a3, v26, (__int64)v30);
    }
    else
    {
      v28 = 0;
    }
    v17 = *(_QWORD *)(a1 + 8);
    if ( v17 )
    {
      v18 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v17 + 40, (__int64)v7);
      v19 = *((_QWORD *)v7 + 3);
      v20 = *v18;
      *((_QWORD *)v7 + 4) = v18;
      v20 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v7 + 3) = v20 | v19 & 7;
      *(_QWORD *)(v20 + 8) = v7 + 24;
      *v18 = *v18 & 7 | (unsigned __int64)(v7 + 24);
    }
    sub_164B780(v28, a5);
    v22 = *(_QWORD *)(a1 + 80) == 0;
    v29 = v7;
    if ( v22 )
      sub_4263D6(v28, a5, v21);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v29);
    v23 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v29 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v29, v23, 2);
      v24 = *((_QWORD *)v7 + 6);
      if ( v24 )
        sub_161E7C0((__int64)(v7 + 48), v24);
      v25 = v29;
      *((_QWORD *)v7 + 6) = v29;
      if ( v25 )
        sub_1623210((__int64)&v29, v25, (__int64)(v7 + 48));
    }
  }
  else
  {
    v7 = (unsigned __int8 *)sub_15A3AE0((_QWORD *)a2, a3, a4, 0);
    v8 = sub_14DBA30((__int64)v7, *(_QWORD *)(a1 + 96), 0);
    if ( v8 )
      return (unsigned __int8 *)v8;
  }
  return v7;
}
