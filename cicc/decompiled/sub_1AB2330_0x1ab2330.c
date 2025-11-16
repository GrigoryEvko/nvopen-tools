// Function: sub_1AB2330
// Address: 0x1ab2330
//
__int64 __fastcall sub_1AB2330(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r15
  __int64 **v11; // rax
  __int64 **v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v20; // [rsp+0h] [rbp-D0h]
  __int64 *v21; // [rsp+8h] [rbp-C8h]
  __int64 v22; // [rsp+10h] [rbp-C0h]
  __int64 v23; // [rsp+18h] [rbp-B8h]
  __int64 v24; // [rsp+18h] [rbp-B8h]
  __int64 v25; // [rsp+20h] [rbp-B0h]
  __int64 v26; // [rsp+20h] [rbp-B0h]
  __int64 v29; // [rsp+48h] [rbp-88h] BYREF
  char v30[16]; // [rsp+50h] [rbp-80h] BYREF
  __int16 v31; // [rsp+60h] [rbp-70h]
  __int64 *v32; // [rsp+70h] [rbp-60h] BYREF
  __int64 v33; // [rsp+78h] [rbp-58h]
  __int64 v34; // [rsp+80h] [rbp-50h] BYREF
  __int64 v35; // [rsp+88h] [rbp-48h]
  __int64 v36; // [rsp+90h] [rbp-40h]
  __int64 v37; // [rsp+98h] [rbp-38h]

  v7 = 0;
  if ( (*(_BYTE *)(*(_QWORD *)a7 + 23LL) & 0x30) != 0 )
  {
    v11 = (__int64 **)sub_157EB90(*(_QWORD *)(a5 + 8));
    LODWORD(v32) = 30;
    v29 = 0;
    v12 = v11;
    v13 = sub_1560040(*v11, -1, (unsigned int *)&v32, 1);
    v14 = *(_QWORD *)(a5 + 8);
    v29 = v13;
    v23 = sub_157E9C0(v14);
    v25 = sub_15A9620(a6, v23, 0);
    v24 = sub_15A9620(a6, v23, 0);
    v15 = sub_16471D0(*(_QWORD **)(a5 + 24), 0);
    v20 = sub_16471D0(*(_QWORD **)(a5 + 24), 0);
    v21 = (__int64 *)sub_16471D0(*(_QWORD **)(a5 + 24), 0);
    v22 = sub_1563520(*v12, &v29, 1);
    v34 = v20;
    v37 = v25;
    v36 = v24;
    v32 = &v34;
    v35 = v15;
    v33 = 0x400000004LL;
    v16 = sub_1644EA0(v21, &v34, 4, 0);
    v17 = sub_1632080((__int64)v12, (__int64)"__memcpy_chk", 12, v16, v22);
    if ( v32 != &v34 )
      _libc_free((unsigned __int64)v32);
    v26 = sub_1AB1800(a1, (__int64 *)a5);
    v33 = sub_1AB1800(a2, (__int64 *)a5);
    v31 = 257;
    v34 = a3;
    v32 = (__int64 *)v26;
    v35 = a4;
    v7 = sub_1285290((__int64 *)a5, *(_QWORD *)(*(_QWORD *)v17 + 24LL), v17, (int)&v32, 4, (__int64)v30, 0);
    v18 = sub_1649C60(v17);
    if ( !*(_BYTE *)(v18 + 16) )
      *(_WORD *)(v7 + 18) = *(_WORD *)(v7 + 18) & 0x8000 | *(_WORD *)(v7 + 18) & 3 | (*(_WORD *)(v18 + 18) >> 2) & 0xFFC;
  }
  return v7;
}
