// Function: sub_136C3D0
// Address: 0x136c3d0
//
__int64 __fastcall sub_136C3D0(__int64 a1, __int64 a2, int a3, __int64 a4, unsigned int a5)
{
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rdi
  _QWORD *v19; // rdx
  __int64 v20; // rdi
  _WORD *v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // rdi
  _WORD *v24; // rdx
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 v27; // rdi
  _BYTE *v28; // rax
  __int64 *v29; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v30; // [rsp+0h] [rbp-D0h]
  __int64 v31; // [rsp+8h] [rbp-C8h]
  unsigned int v32; // [rsp+8h] [rbp-C8h]
  int v33; // [rsp+10h] [rbp-C0h]
  unsigned int v35; // [rsp+2Ch] [rbp-A4h] BYREF
  _QWORD v36[4]; // [rsp+30h] [rbp-A0h] BYREF
  const char *v37; // [rsp+50h] [rbp-80h] BYREF
  __int64 v38; // [rsp+58h] [rbp-78h]
  _QWORD v39[2]; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v40; // [rsp+70h] [rbp-60h] BYREF
  __int64 v41; // [rsp+78h] [rbp-58h]
  __int64 v42; // [rsp+80h] [rbp-50h] BYREF
  __int64 v43; // [rsp+88h] [rbp-48h]
  int v44; // [rsp+90h] [rbp-40h]
  const char **v45; // [rsp+98h] [rbp-38h]

  result = sub_15F4DF0(a4, a5);
  if ( result )
  {
    v9 = result;
    v40 = &v42;
    sub_1367D20((__int64 *)&v40, byte_3F871B3, (__int64)byte_3F871B3);
    v10 = v41;
    if ( v40 != &v42 )
    {
      v31 = v41;
      j_j___libc_free_0(v40, v42 + 1);
      v10 = v31;
    }
    v11 = **(__int64 ***)(a1 + 8);
    v29 = **(__int64 ***)(a1 + 8);
    v32 = qword_4F982C0[20];
    if ( v10 )
    {
      result = sub_1368D10(v11);
      v13 = a5;
      v38 = 0;
      v37 = (const char *)v39;
      v14 = result;
      LOBYTE(v39[0]) = 0;
      if ( !result )
        goto LABEL_11;
    }
    else
    {
      v12 = sub_1368D10(v11);
      v13 = a5;
      v38 = 0;
      v37 = (const char *)v39;
      v14 = v12;
      LOBYTE(v39[0]) = 0;
      a3 = -1;
      if ( !v12 )
        goto LABEL_17;
    }
    v33 = sub_13774A0(v14, a2, a4, v13);
    v44 = 1;
    v43 = 0;
    v42 = 0;
    v40 = (__int64 *)&unk_49EFBE0;
    v45 = &v37;
    v36[1] = "label=\"%.1f%%\"";
    v41 = 0;
    *(double *)&v36[2] = (double)v33 * 100.0 * 4.656612873077393e-10;
    v36[0] = &unk_49E8778;
    sub_16E8450(&v40, v36);
    if ( v32 )
    {
      v36[0] = sub_1368AA0(v29, a2);
      v30 = sub_16AF500(v36, (unsigned int)v33);
      sub_16AF710(&v35, v32, 100);
      v36[0] = *(_QWORD *)(a1 + 24);
      if ( sub_16AF500(v36, v35) <= v30 )
      {
        v22 = (_QWORD *)v43;
        if ( (unsigned __int64)(v42 - v43) > 0xB )
        {
          *(_DWORD *)(v43 + 8) = 577004914;
          *v22 = 0x223D726F6C6F632CLL;
          v15 = v43 + 12;
          v43 += 12;
          goto LABEL_8;
        }
        sub_16E7EE0(&v40, ",color=\"red\"", 12);
      }
    }
    v15 = v43;
LABEL_8:
    if ( v41 != v15 )
      sub_16E7BA0(&v40);
    result = sub_16E7BC0(&v40);
LABEL_11:
    if ( a3 > 64 )
    {
LABEL_12:
      if ( v37 != (const char *)v39 )
        return j_j___libc_free_0(v37, v39[0] + 1LL);
      return result;
    }
LABEL_17:
    v16 = *(_QWORD *)a1;
    v17 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
    if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v17) <= 4 )
    {
      v16 = sub_16E7EE0(v16, "\tNode", 5, v13);
    }
    else
    {
      *(_DWORD *)v17 = 1685016073;
      *(_BYTE *)(v17 + 4) = 101;
      *(_QWORD *)(v16 + 24) += 5LL;
    }
    sub_16E7B40(v16, a2);
    if ( a3 >= 0 )
    {
      v23 = *(_QWORD *)a1;
      v24 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
      if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v24 <= 1u )
      {
        v23 = sub_16E7EE0(v23, ":s", 2);
      }
      else
      {
        *v24 = 29498;
        *(_QWORD *)(v23 + 24) += 2LL;
      }
      sub_16E7AB0(v23, a3);
    }
    v18 = *(_QWORD *)a1;
    v19 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v19 <= 7u )
    {
      v18 = sub_16E7EE0(v18, " -> Node", 8);
    }
    else
    {
      *v19 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v18 + 24) += 8LL;
    }
    sub_16E7B40(v18, v9);
    if ( v38 )
    {
      v25 = *(_QWORD *)a1;
      v26 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v26 )
      {
        v25 = sub_16E7EE0(v25, "[", 1);
      }
      else
      {
        *v26 = 91;
        ++*(_QWORD *)(v25 + 24);
      }
      v27 = sub_16E7EE0(v25, v37, v38);
      v28 = *(_BYTE **)(v27 + 24);
      if ( *(_BYTE **)(v27 + 16) == v28 )
      {
        sub_16E7EE0(v27, "]", 1);
      }
      else
      {
        *v28 = 93;
        ++*(_QWORD *)(v27 + 24);
      }
    }
    v20 = *(_QWORD *)a1;
    v21 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v21 <= 1u )
    {
      result = sub_16E7EE0(v20, ";\n", 2);
    }
    else
    {
      result = 2619;
      *v21 = 2619;
      *(_QWORD *)(v20 + 24) += 2LL;
    }
    goto LABEL_12;
  }
  return result;
}
