// Function: sub_1DDF400
// Address: 0x1ddf400
//
void __fastcall sub_1DDF400(__int64 a1, __int64 a2, int a3, __int64 *a4)
{
  __int64 v4; // r14
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  _WORD *v19; // rdx
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  _WORD *v22; // rdx
  __int64 v23; // rdi
  _BYTE *v24; // rax
  __int64 v25; // rdi
  _BYTE *v26; // rax
  int v27; // [rsp+4h] [rbp-CCh]
  __int64 v28; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v29; // [rsp+8h] [rbp-C8h]
  __int64 v30; // [rsp+10h] [rbp-C0h]
  unsigned int v31; // [rsp+18h] [rbp-B8h]
  int v33; // [rsp+2Ch] [rbp-A4h] BYREF
  __int64 v34[4]; // [rsp+30h] [rbp-A0h] BYREF
  char *v35; // [rsp+50h] [rbp-80h] BYREF
  size_t v36; // [rsp+58h] [rbp-78h]
  _QWORD v37[2]; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v38; // [rsp+70h] [rbp-60h] BYREF
  __int64 v39; // [rsp+78h] [rbp-58h]
  __int64 v40; // [rsp+80h] [rbp-50h] BYREF
  __int64 v41; // [rsp+88h] [rbp-48h]
  int v42; // [rsp+90h] [rbp-40h]
  char **v43; // [rsp+98h] [rbp-38h]

  v4 = *a4;
  if ( *a4 )
  {
    v38 = &v40;
    sub_1DDBCE0((__int64 *)&v38, byte_3F871B3, (__int64)byte_3F871B3);
    v7 = v39;
    if ( v38 != &v40 )
    {
      v30 = v39;
      j_j___libc_free_0(v38, v40 + 1);
      v7 = v30;
    }
    v28 = **(_QWORD **)(a1 + 8);
    v31 = qword_4F982C0[20];
    if ( v7 )
    {
      v36 = 0;
      v8 = sub_1DDC530(v28);
      LOBYTE(v37[0]) = 0;
      v35 = (char *)v37;
      if ( !v8 )
        goto LABEL_11;
    }
    else
    {
      v36 = 0;
      v8 = sub_1DDC530(v28);
      LOBYTE(v37[0]) = 0;
      v35 = (char *)v37;
      a3 = -1;
      if ( !v8 )
        goto LABEL_17;
    }
    v27 = sub_1DF1770(v8, a2, a4);
    v42 = 1;
    v41 = 0;
    v40 = 0;
    v38 = (__int64 *)&unk_49EFBE0;
    v43 = &v35;
    v34[1] = (__int64)"label=\"%.1f%%\"";
    v39 = 0;
    *(double *)&v34[2] = (double)v27 * 100.0 * 4.656612873077393e-10;
    v34[0] = (__int64)&unk_49E8778;
    sub_16E8450((__int64)&v38, (__int64)v34, v9, v10, v11, v12);
    if ( v31 )
    {
      v34[0] = sub_1DDC3C0(v28, a2);
      v29 = sub_16AF500(v34, v27);
      sub_16AF710(&v33, v31, 0x64u);
      v34[0] = *(_QWORD *)(a1 + 24);
      if ( sub_16AF500(v34, v33) <= v29 )
      {
        v20 = (_QWORD *)v41;
        if ( (unsigned __int64)(v40 - v41) > 0xB )
        {
          *(_DWORD *)(v41 + 8) = 577004914;
          *v20 = 0x223D726F6C6F632CLL;
          v13 = v41 + 12;
          v41 += 12;
          goto LABEL_8;
        }
        sub_16E7EE0((__int64)&v38, ",color=\"red\"", 0xCu);
      }
    }
    v13 = v41;
LABEL_8:
    if ( v39 != v13 )
      sub_16E7BA0((__int64 *)&v38);
    sub_16E7BC0((__int64 *)&v38);
LABEL_11:
    if ( a3 > 64 )
    {
LABEL_12:
      if ( v35 != (char *)v37 )
        j_j___libc_free_0(v35, v37[0] + 1LL);
      return;
    }
LABEL_17:
    v14 = *(_QWORD *)a1;
    v15 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
    if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v15) <= 4 )
    {
      v14 = sub_16E7EE0(v14, "\tNode", 5u);
    }
    else
    {
      *(_DWORD *)v15 = 1685016073;
      *(_BYTE *)(v15 + 4) = 101;
      *(_QWORD *)(v14 + 24) += 5LL;
    }
    sub_16E7B40(v14, a2);
    if ( a3 >= 0 )
    {
      v21 = *(_QWORD *)a1;
      v22 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
      if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v22 <= 1u )
      {
        v21 = sub_16E7EE0(v21, ":s", 2u);
      }
      else
      {
        *v22 = 29498;
        *(_QWORD *)(v21 + 24) += 2LL;
      }
      sub_16E7AB0(v21, a3);
    }
    v16 = *(_QWORD *)a1;
    v17 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v17 <= 7u )
    {
      v16 = sub_16E7EE0(v16, " -> Node", 8u);
    }
    else
    {
      *v17 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v16 + 24) += 8LL;
    }
    sub_16E7B40(v16, v4);
    if ( v36 )
    {
      v23 = *(_QWORD *)a1;
      v24 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v24 )
      {
        v23 = sub_16E7EE0(v23, "[", 1u);
      }
      else
      {
        *v24 = 91;
        ++*(_QWORD *)(v23 + 24);
      }
      v25 = sub_16E7EE0(v23, v35, v36);
      v26 = *(_BYTE **)(v25 + 24);
      if ( *(_BYTE **)(v25 + 16) == v26 )
      {
        sub_16E7EE0(v25, "]", 1u);
      }
      else
      {
        *v26 = 93;
        ++*(_QWORD *)(v25 + 24);
      }
    }
    v18 = *(_QWORD *)a1;
    v19 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v19 <= 1u )
    {
      sub_16E7EE0(v18, ";\n", 2u);
    }
    else
    {
      *v19 = 2619;
      *(_QWORD *)(v18 + 24) += 2LL;
    }
    goto LABEL_12;
  }
}
