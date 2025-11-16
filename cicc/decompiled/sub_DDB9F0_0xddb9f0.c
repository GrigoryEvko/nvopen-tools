// Function: sub_DDB9F0
// Address: 0xddb9f0
//
char __fastcall sub_DDB9F0(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        _BYTE *a4,
        unsigned __int64 a5,
        _BYTE *a6,
        _BYTE *a7,
        __int64 a8)
{
  _BYTE *v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned int v17; // eax
  unsigned __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *v20; // rbx
  _BYTE *v21; // rbx
  _BYTE *v22; // rax
  char result; // al
  __int64 v24; // rax
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  _BYTE *v30; // r14
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // [rsp+8h] [rbp-68h]
  _BYTE *v35; // [rsp+20h] [rbp-50h]
  unsigned __int64 v37; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v38; // [rsp+38h] [rbp-38h]

  v10 = (_BYTE *)a3;
  v11 = sub_D95540(a3);
  v12 = sub_D97050((__int64)a1, v11);
  v13 = sub_D95540((__int64)a6);
  if ( v12 >= sub_D97050((__int64)a1, v13) )
  {
    v24 = sub_D95540((__int64)v10);
    v25 = sub_D97050((__int64)a1, v24);
    v26 = sub_D95540((__int64)a6);
    if ( v25 <= sub_D97050((__int64)a1, v26) )
      return sub_DDB1C0(a1, a2, v10, a4, a5, a6, a7, a8);
    if ( *(_BYTE *)(sub_D95540((__int64)a6) + 8) != 14 && *(_BYTE *)(sub_D95540((__int64)a7) + 8) != 14 )
    {
      if ( sub_B532B0(a5) )
      {
        v29 = sub_D95540((__int64)v10);
        v30 = sub_DC5000((__int64)a1, (__int64)a6, v29, 0);
        v31 = sub_D95540((__int64)v10);
        v35 = sub_DC5000((__int64)a1, (__int64)a7, v31, 0);
      }
      else
      {
        v32 = sub_D95540((__int64)v10);
        v30 = sub_DC2B70((__int64)a1, (__int64)a6, v32, 0);
        v33 = sub_D95540((__int64)v10);
        v35 = sub_DC2B70((__int64)a1, (__int64)a7, v33, 0);
      }
      return sub_DDB1C0(a1, a2, v10, a4, a5, v30, v35, a8);
    }
    return 0;
  }
  if ( sub_B532B0(a5) || *(_BYTE *)(sub_D95540((__int64)a6) + 8) == 14 || *(_BYTE *)(sub_D95540((__int64)a7) + 8) == 14 )
    goto LABEL_4;
  v34 = sub_D95540((__int64)v10);
  v16 = sub_D95540((__int64)a6);
  v17 = sub_D97050((__int64)a1, v34);
  v38 = v17;
  if ( v17 > 0x40 )
  {
    sub_C43690((__int64)&v37, -1, 1);
  }
  else
  {
    v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
    if ( !v17 )
      v18 = 0;
    v37 = v18;
  }
  v19 = sub_DA26C0(a1, (__int64)&v37);
  v20 = sub_DC2B70((__int64)a1, (__int64)v19, v16, 0);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( !(unsigned __int8)sub_DCD020(a1, 37, (__int64)a6, (__int64)v20)
    || !(unsigned __int8)sub_DCD020(a1, 37, (__int64)a7, (__int64)v20)
    || (v21 = sub_DC5200((__int64)a1, (__int64)a6, v34, 0),
        v22 = sub_DC5200((__int64)a1, (__int64)a7, v34, 0),
        (result = sub_DDB1C0(a1, a2, v10, a4, a5, v21, v22, a8)) == 0) )
  {
LABEL_4:
    if ( *(_BYTE *)(sub_D95540((__int64)v10) + 8) != 14 && *(_BYTE *)(sub_D95540((__int64)a4) + 8) != 14 )
    {
      if ( sub_B532B0(a2) )
      {
        v14 = sub_D95540((__int64)a6);
        v10 = sub_DC5000((__int64)a1, (__int64)v10, v14, 0);
        v15 = sub_D95540((__int64)a6);
        a4 = sub_DC5000((__int64)a1, (__int64)a4, v15, 0);
      }
      else
      {
        v27 = sub_D95540((__int64)a6);
        v10 = sub_DC2B70((__int64)a1, (__int64)v10, v27, 0);
        v28 = sub_D95540((__int64)a6);
        a4 = sub_DC2B70((__int64)a1, (__int64)a4, v28, 0);
      }
      return sub_DDB1C0(a1, a2, v10, a4, a5, a6, a7, a8);
    }
    return 0;
  }
  return result;
}
