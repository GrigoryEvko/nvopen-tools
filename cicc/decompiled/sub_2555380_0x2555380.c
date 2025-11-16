// Function: sub_2555380
// Address: 0x2555380
//
__int64 __fastcall sub_2555380(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  unsigned int v3; // r12d
  unsigned int v5; // eax
  bool v6; // bl
  char v7; // al
  __int64 v8; // rbx
  unsigned __int8 v9; // al
  _QWORD *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 *v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-88h]
  _QWORD *v21; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-78h]
  _QWORD *v23; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-68h]
  _QWORD *v25; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+48h] [rbp-58h]
  _QWORD *v27; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v28; // [rsp+58h] [rbp-48h]
  _QWORD *v29; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+68h] [rbp-38h]

  sub_25550E0((__int64)&v23, a1, a2, 0);
  v2 = sub_250D070((_QWORD *)(a1 + 72));
  if ( sub_AAF7D0((__int64)&v23) )
    goto LABEL_2;
  v22 = v24;
  if ( v24 > 0x40 )
    sub_C43780((__int64)&v21, (const void **)&v23);
  else
    v21 = v23;
  sub_C46A40((__int64)&v21, 1);
  v5 = v22;
  v22 = 0;
  LODWORD(v28) = v5;
  v27 = v21;
  v6 = v26 <= 0x40 ? v25 == v21 : sub_C43C50((__int64)&v25, (const void **)&v27);
  sub_969240((__int64 *)&v27);
  sub_969240((__int64 *)&v21);
  if ( v6 )
    goto LABEL_2;
  v7 = *(_BYTE *)v2;
  if ( *(_BYTE *)v2 <= 0x1Cu || v7 != 85 && v7 != 61 )
    goto LABEL_2;
  if ( (*(_BYTE *)(v2 + 7) & 0x20) == 0 )
  {
    if ( sub_AAF760((__int64)&v23) )
      goto LABEL_2;
    goto LABEL_32;
  }
  v8 = sub_B91C10(v2, 4);
  if ( sub_AAF760((__int64)&v23) )
    goto LABEL_2;
  if ( v8 )
  {
    v9 = *(_BYTE *)(v8 - 16);
    if ( (v9 & 2) != 0 )
    {
      if ( *(_DWORD *)(v8 - 24) > 2u )
        goto LABEL_2;
      v10 = *(_QWORD **)(v8 - 32);
      v11 = *(_QWORD *)(*v10 + 136LL);
    }
    else
    {
      if ( ((*(_WORD *)(v8 - 16) >> 6) & 0xFu) > 2 )
        goto LABEL_2;
      v10 = (_QWORD *)(v8 - 8LL * ((v9 >> 2) & 0xF) - 16);
      v11 = *(_QWORD *)(*v10 + 136LL);
    }
    v12 = *(_QWORD *)(v10[1] + 136LL);
    v22 = *(_DWORD *)(v12 + 32);
    if ( v22 > 0x40 )
      sub_C43780((__int64)&v21, (const void **)(v12 + 24));
    else
      v21 = *(_QWORD **)(v12 + 24);
    v20 = *(_DWORD *)(v11 + 32);
    if ( v20 > 0x40 )
      sub_C43780((__int64)&v19, (const void **)(v11 + 24));
    else
      v19 = *(_QWORD *)(v11 + 24);
    sub_AADC30((__int64)&v27, (__int64)&v19, (__int64 *)&v21);
    sub_969240(&v19);
    sub_969240((__int64 *)&v21);
    if ( (unsigned __int8)sub_AB1BB0((__int64)&v27, (__int64)&v23) )
    {
      if ( (unsigned int)v28 <= 0x40 )
      {
        if ( v27 != v23 )
          goto LABEL_31;
      }
      else if ( !sub_C43C50((__int64)&v27, (const void **)&v23) )
      {
LABEL_31:
        sub_969240((__int64 *)&v29);
        sub_969240((__int64 *)&v27);
        goto LABEL_32;
      }
      if ( v30 <= 0x40 )
      {
        if ( v29 != v25 )
          goto LABEL_31;
      }
      else if ( !sub_C43C50((__int64)&v29, (const void **)&v25) )
      {
        goto LABEL_31;
      }
    }
    sub_969240((__int64 *)&v29);
    sub_969240((__int64 *)&v27);
    goto LABEL_2;
  }
LABEL_32:
  if ( !sub_AAF7D0((__int64)&v23) )
  {
    v13 = sub_BD5C60(v2);
    v14 = *(_QWORD *)(v2 + 8);
    v3 = 0;
    v15 = (__int64 *)v13;
    v16 = sub_AD8D80(v14, (__int64)&v23);
    v27 = sub_B98A20(v16, (__int64)&v23);
    v17 = sub_AD8D80(v14, (__int64)&v25);
    v28 = sub_B98A20(v17, (__int64)&v25);
    v18 = sub_B9C770(v15, (__int64 *)&v27, (__int64 *)2, 0, 1);
    sub_B99FD0(v2, 4u, v18);
    goto LABEL_3;
  }
LABEL_2:
  v3 = 1;
LABEL_3:
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0((unsigned __int64)v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0((unsigned __int64)v23);
  return v3;
}
