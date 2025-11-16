// Function: sub_DE4FD0
// Address: 0xde4fd0
//
__int64 __fastcall sub_DE4FD0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  unsigned int v6; // r15d
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rbx
  bool v22; // al
  __int64 v23; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v24; // [rsp+18h] [rbp-A8h]
  _QWORD *v25; // [rsp+18h] [rbp-A8h]
  _QWORD *v28; // [rsp+28h] [rbp-98h]
  _QWORD *v29; // [rsp+28h] [rbp-98h]
  const void *v30; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-88h]
  __int64 v32; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v33; // [rsp+48h] [rbp-78h]
  const void *v34; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v35; // [rsp+58h] [rbp-68h]
  __int64 v36; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v37; // [rsp+68h] [rbp-58h]
  __int64 v38; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+78h] [rbp-48h]
  __int64 v40; // [rsp+80h] [rbp-40h]
  unsigned int v41; // [rsp+88h] [rbp-38h]

  v6 = a5;
  v8 = sub_D33D80((_QWORD *)a3, (__int64)a2, a3, a4, a5);
  if ( *(_WORD *)(v8 + 24) )
    goto LABEL_3;
  v9 = v8;
  v10 = sub_D95540(a4);
  v24 = sub_D97050((__int64)a2, v10);
  v11 = sub_D95540(**(_QWORD **)(a3 + 32));
  if ( v24 > sub_D97050((__int64)a2, v11)
    || (v13 = sub_D95540(**(_QWORD **)(a3 + 32)),
        v28 = sub_DC2CB0((__int64)a2, a4, v13),
        v14 = sub_D95540(**(_QWORD **)(a3 + 32)),
        v25 = sub_DA2C50((__int64)a2, v14, -1, 1u),
        v15 = sub_DCAF50(a2, v9, 0),
        v16 = sub_DCEE80(a2, v9, (__int64)v15, 0),
        v17 = sub_DCB270((__int64)a2, (__int64)v25, (__int64)v16),
        !(unsigned __int8)sub_DCCA40(a2, 37, (__int64)v28, v17)) )
  {
LABEL_3:
    sub_AADB10(a1, v6, 1);
  }
  else
  {
    v29 = sub_DD0540(a3, (__int64)v28, a2);
    v23 = sub_DE4F70(a2, **(_QWORD **)(a3 + 32), *(_QWORD *)(a3 + 48));
    v18 = sub_DBB9F0((__int64)a2, v23, a6, 0);
    v19 = v18;
    v31 = *(_DWORD *)(v18 + 8);
    if ( v31 > 0x40 )
      sub_C43780((__int64)&v30, (const void **)v18);
    else
      v30 = *(const void **)v18;
    v33 = *(_DWORD *)(v19 + 24);
    if ( v33 > 0x40 )
      sub_C43780((__int64)&v32, (const void **)(v19 + 16));
    else
      v32 = *(_QWORD *)(v19 + 16);
    v20 = sub_DBB9F0((__int64)a2, (__int64)v29, a6, 0);
    v21 = v20;
    v35 = *(_DWORD *)(v20 + 8);
    if ( v35 > 0x40 )
      sub_C43780((__int64)&v34, (const void **)v20);
    else
      v34 = *(const void **)v20;
    v37 = *(_DWORD *)(v21 + 24);
    if ( v37 > 0x40 )
      sub_C43780((__int64)&v36, (const void **)(v21 + 16));
    else
      v36 = *(_QWORD *)(v21 + 16);
    sub_AB3510((__int64)&v38, (__int64)&v30, (__int64)&v34, 0);
    if ( sub_AAF760((__int64)&v38)
      || (a6 == 1 ? (v22 = sub_AB0120((__int64)&v38)) : (v22 = sub_AAFBB0((__int64)&v38)),
          !v22
       && ((unsigned __int8)sub_DBEDC0((__int64)a2, v9)
        && (unsigned __int8)sub_DCCA40(a2, (4 * (a6 == 1) + 37) & 0x7F, v23, (__int64)v29)
        || (unsigned __int8)sub_DBEC00((__int64)a2, v9)
        && (unsigned __int8)sub_DCCA40(a2, (4 * (a6 == 1) + 35) & 0x7F, v23, (__int64)v29))) )
    {
      *(_DWORD *)(a1 + 8) = v39;
      *(_QWORD *)a1 = v38;
      *(_DWORD *)(a1 + 24) = v41;
      *(_QWORD *)(a1 + 16) = v40;
    }
    else
    {
      sub_AADB10(a1, v6, 1);
      if ( v41 > 0x40 && v40 )
        j_j___libc_free_0_0(v40);
      if ( v39 > 0x40 && v38 )
        j_j___libc_free_0_0(v38);
    }
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    if ( v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
  }
  return a1;
}
