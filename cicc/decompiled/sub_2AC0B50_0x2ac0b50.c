// Function: sub_2AC0B50
// Address: 0x2ac0b50
//
__int64 __fastcall sub_2AC0B50(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v4; // r12
  unsigned __int8 *v5; // r13
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  unsigned __int64 v13; // r15
  signed __int64 v14; // rcx
  bool v15; // of
  signed __int64 v16; // r15
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 *v19; // r11
  __int64 v20; // r10
  unsigned __int64 v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // r15
  __int64 v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // r15
  __int64 v29; // r12
  __int64 v30; // r14
  __int64 *v31; // rax
  __int64 v32; // [rsp+0h] [rbp-A0h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  signed __int64 v34; // [rsp+28h] [rbp-78h]
  __int64 v35; // [rsp+30h] [rbp-70h]
  __int64 v36; // [rsp+30h] [rbp-70h]
  _QWORD **v37; // [rsp+40h] [rbp-60h]
  __int64 *v38; // [rsp+40h] [rbp-60h]
  __int64 v40; // [rsp+50h] [rbp-50h] BYREF
  __int64 v41; // [rsp+58h] [rbp-48h]
  signed __int64 v42; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v43; // [rsp+68h] [rbp-38h]

  if ( BYTE4(a3) )
    return 0;
  v4 = (__int64)a1;
  v5 = (unsigned __int8 *)a2;
  if ( *(_BYTE *)a2 == 61 )
    v37 = *(_QWORD ***)(a2 + 8);
  else
    v37 = *(_QWORD ***)(*(_QWORD *)(a2 - 64) + 8LL);
  v6 = sub_228AED0((_BYTE *)a2);
  sub_2AAEDF0(*(_QWORD *)(v6 + 8), a3);
  if ( *(_BYTE *)v6 == 63 )
  {
    v32 = a1[53];
    v25 = *(_QWORD *)(v32 + 112);
    v26 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
    if ( (unsigned int)v26 > 1 )
    {
      v27 = v6;
      v28 = 1;
      v33 = a1[55];
      v29 = a1[52];
      v35 = (unsigned int)(v26 - 1);
      while ( 1 )
      {
        v30 = *(_QWORD *)(v27 + 32 * (v28 - v26));
        v31 = sub_DD8400(v25, v30);
        if ( !sub_DADE90(v25, (__int64)v31, v29) && !(unsigned __int8)sub_31A6B90(v33, v30) )
        {
          v4 = (__int64)a1;
          v5 = (unsigned __int8 *)a2;
          goto LABEL_6;
        }
        if ( v28 == v35 )
          break;
        ++v28;
        v26 = *(_DWORD *)(v27 + 4) & 0x7FFFFFF;
      }
      v6 = v27;
      v4 = (__int64)a1;
      v5 = (unsigned __int8 *)a2;
    }
    sub_DEEF40(v32, v6);
  }
LABEL_6:
  v7 = sub_DFDB90(*(_QWORD *)(v4 + 448));
  v43 = 0;
  v40 = v7;
  v41 = v8;
  v42 = (unsigned int)a3;
  sub_2AA9150((__int64)&v42, (__int64)&v40);
  v34 = v42;
  sub_2AAE0E0((__int64)v5);
  v9 = sub_DFD4A0(*(__int64 **)(v4 + 448));
  v43 = 0;
  v40 = v9;
  v41 = v10;
  v42 = (unsigned int)a3;
  sub_2AA9150((__int64)&v42, (__int64)&v40);
  v13 = v42 + v34;
  if ( __OFADD__(v42, v34) )
  {
    v13 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v42 <= 0 )
      v13 = 0x8000000000000000LL;
  }
  v14 = sub_2AC04F0(v4, (__int64)v5, a3, v11, 0, v12);
  v15 = __OFADD__(v14, v13);
  v16 = v14 + v13;
  if ( v15 )
  {
    v16 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v14 <= 0 )
      v16 = 0x8000000000000000LL;
  }
  if ( (unsigned __int8)sub_2AB37C0(v4, v5) )
  {
    if ( *(_DWORD *)(v4 + 992) != 2 )
      v16 /= 2LL;
    v17 = (__int64 *)sub_BCB2A0(*v37);
    v18 = sub_BCE1B0(v17, a3);
    v19 = *(__int64 **)(v4 + 448);
    v20 = v18;
    v43 = a3;
    if ( (unsigned int)a3 > 0x40 )
    {
      v38 = v19;
      v36 = v18;
      sub_C43690((__int64)&v42, -1, 1);
      v20 = v36;
      v19 = v38;
    }
    else
    {
      v21 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
      if ( !(_DWORD)a3 )
        v21 = 0;
      v42 = v21;
    }
    v22 = sub_DFAAD0(v19, v20, (__int64)&v42, 0, 1u);
    v15 = __OFADD__(v22, v16);
    v23 = v22 + v16;
    if ( v15 )
    {
      v23 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v22 <= 0 )
        v23 = 0x8000000000000000LL;
    }
    if ( v43 > 0x40 && v42 )
      j_j___libc_free_0_0(v42);
    v24 = sub_DFD270(*(_QWORD *)(v4 + 448), 2, *(_DWORD *)(v4 + 992));
    v15 = __OFADD__(v24, v23);
    v16 = v24 + v23;
    if ( v15 )
    {
      v16 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v24 <= 0 )
        v16 = 0x8000000000000000LL;
    }
    if ( sub_2AB4AE0((_DWORD *)v4, v5) )
      return 3000000;
  }
  return v16;
}
