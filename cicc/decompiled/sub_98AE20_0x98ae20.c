// Function: sub_98AE20
// Address: 0x98ae20
//
__int64 __fastcall sub_98AE20(__int64 a1, __int64 *a2, unsigned int a3, __int64 a4)
{
  unsigned __int8 *v6; // rax
  __int64 v7; // r14
  int v8; // eax
  unsigned int v9; // r15d
  unsigned int v11; // eax
  _QWORD *v12; // r8
  unsigned int v13; // r12d
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // [rsp+0h] [rbp-70h]
  _QWORD *v29; // [rsp+0h] [rbp-70h]
  __int64 v30; // [rsp+8h] [rbp-68h]
  unsigned __int8 v31; // [rsp+10h] [rbp-60h]
  unsigned __int64 v32; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-48h]

  v6 = sub_98ACB0((unsigned __int8 *)a1, 6u);
  if ( *v6 != 3 )
    return 0;
  v7 = (__int64)v6;
  v8 = v6[80] & 1;
  v31 = v8;
  if ( !v8 )
    return 0;
  if ( (unsigned __int8)sub_B2FC80(v7) )
    return 0;
  if ( (unsigned __int8)sub_B2F6B0(v7) )
    return 0;
  v9 = (*(_BYTE *)(v7 + 80) & 2) != 0;
  if ( (*(_BYTE *)(v7 + 80) & 2) != 0 )
    return 0;
  v30 = sub_B2F730(v7);
  v33 = sub_AE43F0(v30, *(_QWORD *)(a1 + 8));
  if ( v33 > 0x40 )
    sub_C43690(&v32, 0, 0);
  else
    v32 = 0;
  if ( v7 != sub_BD45C0(a1, v30, (unsigned int)&v32, 1, 0, 0, 0, 0) )
    goto LABEL_11;
  v13 = v33;
  if ( v33 <= 0x40 )
  {
    v14 = v32;
    if ( v32 == -1 )
      return v9;
    v15 = a3 >> 3;
    if ( v32 % v15 )
      return v9;
    goto LABEL_19;
  }
  v29 = (_QWORD *)v32;
  v22 = sub_C444A0(&v32);
  v12 = v29;
  if ( v13 - v22 <= 0x40 )
  {
    v14 = *v29;
    if ( *v29 == -1 )
      goto LABEL_15;
    v15 = a3 >> 3;
    if ( v14 % v15 )
    {
LABEL_13:
      v12 = (_QWORD *)v32;
      goto LABEL_14;
    }
LABEL_19:
    v28 = v15;
    v16 = a4 + v14 / v15;
    v9 = sub_AC30F0(*(_QWORD *)(v7 - 32));
    if ( (_BYTE)v9 )
    {
      v25 = sub_9208B0(v30, *(_QWORD *)(v7 + 24));
      *a2 = 0;
      v26 = ((unsigned __int64)(v25 + 7) >> 3) / v28;
      a2[1] = 0;
      v27 = v26 - v16;
      if ( v16 > v26 )
        v27 = 0;
      v11 = v33;
      a2[2] = v27;
LABEL_12:
      if ( v11 <= 0x40 )
        return v9;
      goto LABEL_13;
    }
    v18 = *(_QWORD *)(v7 - 32);
    if ( *(_BYTE *)v18 == 15
      && (v19 = sub_AC5230(*(_QWORD *)(v7 - 32), v30, v17, v28), (unsigned __int8)sub_BCAC40(v19, a3)) )
    {
      v20 = *(_QWORD *)(*(_QWORD *)(v18 + 8) + 32LL);
      if ( v16 <= v20 )
      {
        v21 = v20 - v16;
LABEL_24:
        v9 = v31;
        a2[2] = v21;
        v11 = v33;
        *a2 = v18;
        a2[1] = v16;
        goto LABEL_12;
      }
    }
    else if ( a3 == 8 )
    {
      v23 = sub_970A00(v7, v16);
      if ( v23 )
      {
        v18 = 0;
        if ( *(_BYTE *)v23 == 15 )
          v18 = v23;
        v24 = *(_QWORD *)(v23 + 8);
        if ( *(_BYTE *)(v24 + 8) != 16 )
          BUG();
        v21 = *(_QWORD *)(v24 + 32);
        v16 = 0;
        goto LABEL_24;
      }
    }
LABEL_11:
    v11 = v33;
    goto LABEL_12;
  }
LABEL_14:
  if ( v12 )
LABEL_15:
    j_j___libc_free_0_0(v12);
  return v9;
}
