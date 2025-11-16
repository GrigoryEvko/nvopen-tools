// Function: sub_DD0090
// Address: 0xdd0090
//
_QWORD *__fastcall sub_DD0090(_QWORD **a1, int a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rbx
  __int64 *v7; // rbx
  __int64 v8; // rax
  unsigned int v9; // eax
  unsigned int v11; // ebx
  unsigned int v12; // r14d
  unsigned int v13; // r13d
  __int64 v15; // r13
  unsigned int v16; // r8d
  unsigned int v17; // ebx
  _QWORD *v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rbx
  __int64 v27; // [rsp+18h] [rbp-A8h]
  _QWORD *v28; // [rsp+20h] [rbp-A0h]
  __int64 v29; // [rsp+28h] [rbp-98h]
  __int64 v30; // [rsp+28h] [rbp-98h]
  unsigned int v31; // [rsp+30h] [rbp-90h]
  unsigned int v32; // [rsp+30h] [rbp-90h]
  _QWORD *v33; // [rsp+30h] [rbp-90h]
  unsigned int v34; // [rsp+3Ch] [rbp-84h]
  __int64 v35; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-78h]
  __int64 v37; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+58h] [rbp-68h]
  __int64 v39; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+68h] [rbp-58h]
  _QWORD *v41; // [rsp+70h] [rbp-50h] BYREF
  __int64 v42; // [rsp+78h] [rbp-48h]
  _QWORD *v43; // [rsp+80h] [rbp-40h] BYREF
  _QWORD *v44; // [rsp+88h] [rbp-38h]

  v28 = *a1;
  if ( a2 == 1 )
    return v28;
  v34 = 1;
  while ( 1 )
  {
    v8 = sub_D95540((__int64)v28);
    v27 = v8;
    if ( v34 == 1 )
    {
      v6 = (__int64)sub_DC5760((__int64)a4, a3, v8, 0);
      goto LABEL_4;
    }
    if ( v34 > 0x3E8 )
    {
      v6 = sub_D970F0((__int64)a4);
      goto LABEL_4;
    }
    v31 = sub_D97050((__int64)a4, v8);
    v36 = v31;
    if ( v31 > 0x40 )
    {
      sub_C43690((__int64)&v35, 1, 0);
      v9 = v34;
      if ( v34 <= 2 )
      {
LABEL_41:
        v15 = 2;
        v11 = 1;
        goto LABEL_18;
      }
    }
    else
    {
      v9 = v34;
      v35 = 1;
      if ( v34 <= 2 )
        goto LABEL_41;
    }
    v29 = a3;
    _ECX = 0;
    v11 = 1;
    v12 = 3;
    v13 = v9;
    while ( 1 )
    {
      v11 += _ECX;
      sub_C47170((__int64)&v35, v12 >> _ECX);
      _EAX = v12 + 1;
      if ( v12 == v13 )
        break;
      ++v12;
      __asm { tzcnt   ecx, eax }
    }
    a3 = v29;
    v15 = 1LL << v11;
LABEL_18:
    v16 = v11 + v31;
    v38 = v11 + v31;
    if ( v11 + v31 > 0x40 )
    {
      sub_C43690((__int64)&v37, 0, 0);
      v16 = v11 + v31;
      if ( v38 > 0x40 )
      {
        *(_QWORD *)(v37 + 8LL * (v11 >> 6)) |= v15;
        goto LABEL_21;
      }
    }
    else
    {
      v37 = 0;
    }
    v37 |= v15;
LABEL_21:
    v17 = 1;
    v32 = v16;
    sub_C473B0((__int64)&v39, (__int64)&v35);
    v18 = (_QWORD *)sub_B2BE50(*a4);
    v30 = sub_BCCE00(v18, v32);
    v33 = sub_DC5760((__int64)a4, a3, v30, 0);
    do
    {
      v19 = sub_D95540(a3);
      v20 = sub_DA2C50((__int64)a4, v19, v17, 0);
      v21 = sub_DCC810(a4, a3, (__int64)v20, 0, 0);
      v44 = sub_DC5760((__int64)a4, (__int64)v21, v30, 0);
      v41 = &v43;
      v43 = v33;
      v42 = 0x200000002LL;
      v33 = sub_DC8BD0(a4, (__int64)&v41, 0, 0);
      if ( v41 != &v43 )
        _libc_free(v41, &v41);
      ++v17;
    }
    while ( v17 != v34 );
    v22 = sub_DA26C0(a4, (__int64)&v37);
    v23 = sub_DCB270((__int64)a4, (__int64)v33, (__int64)v22);
    v24 = sub_DC5760((__int64)a4, v23, v27, 0);
    v43 = sub_DA26C0(a4, (__int64)&v39);
    v44 = v24;
    v41 = &v43;
    v42 = 0x200000002LL;
    v6 = (__int64)sub_DC8BD0(a4, (__int64)&v41, 0, 0);
    if ( v41 != &v43 )
      _libc_free(v41, &v41);
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    if ( v38 > 0x40 && v37 )
      j_j___libc_free_0_0(v37);
    if ( v36 > 0x40 && v35 )
      break;
LABEL_4:
    if ( sub_D96A50(v6) )
      return (_QWORD *)v6;
LABEL_5:
    v44 = (_QWORD *)v6;
    v41 = &v43;
    v43 = a1[v34];
    v42 = 0x200000002LL;
    v7 = sub_DC8BD0(a4, (__int64)&v41, 0, 0);
    if ( v41 != &v43 )
      _libc_free(v41, &v41);
    v41 = &v43;
    v43 = v28;
    v44 = v7;
    v42 = 0x200000002LL;
    v28 = sub_DC7EB0(a4, (__int64)&v41, 0, 0);
    if ( v41 != &v43 )
      _libc_free(v41, &v41);
    if ( a2 == ++v34 )
      return v28;
  }
  j_j___libc_free_0_0(v35);
  if ( !sub_D96A50(v6) )
    goto LABEL_5;
  return (_QWORD *)v6;
}
