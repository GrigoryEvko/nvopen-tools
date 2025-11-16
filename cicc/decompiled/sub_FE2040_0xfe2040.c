// Function: sub_FE2040
// Address: 0xfe2040
//
__int64 __fastcall sub_FE2040(__int64 a1, unsigned __int64 a2, int a3, __int64 a4, unsigned int a5)
{
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rdi
  _QWORD *v23; // rdx
  __int64 v24; // rdi
  _WORD *v25; // rdx
  unsigned __int64 v26; // r14
  _QWORD *v27; // rdx
  __int64 v28; // rdi
  _WORD *v29; // rdx
  __int64 v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // rdi
  _BYTE *v33; // rax
  int v34; // [rsp+10h] [rbp-E0h]
  unsigned int v35; // [rsp+14h] [rbp-DCh]
  unsigned __int64 v36; // [rsp+18h] [rbp-D8h]
  __int64 v37; // [rsp+20h] [rbp-D0h]
  int v38; // [rsp+20h] [rbp-D0h]
  unsigned int v39; // [rsp+3Ch] [rbp-B4h] BYREF
  _QWORD v40[4]; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int8 *v41; // [rsp+60h] [rbp-90h] BYREF
  size_t v42; // [rsp+68h] [rbp-88h]
  _QWORD v43[2]; // [rsp+70h] [rbp-80h] BYREF
  __int64 *v44; // [rsp+80h] [rbp-70h] BYREF
  __int64 v45; // [rsp+88h] [rbp-68h]
  __int64 v46; // [rsp+90h] [rbp-60h] BYREF
  __int64 v47; // [rsp+98h] [rbp-58h]
  __int64 v48; // [rsp+A0h] [rbp-50h]
  __int64 v49; // [rsp+A8h] [rbp-48h]
  unsigned __int8 **v50; // [rsp+B0h] [rbp-40h]

  result = sub_B46EC0(a4, a5);
  v36 = result;
  if ( result )
  {
    v44 = &v46;
    sub_FDB1F0((__int64 *)&v44, byte_3F871B3, (__int64)byte_3F871B3);
    v10 = v45;
    if ( v44 != &v46 )
    {
      v37 = v45;
      j_j___libc_free_0(v44, v46 + 1);
      v10 = v37;
    }
    if ( v10 )
    {
      v38 = a3;
      v11 = **(__int64 ***)(a1 + 8);
      v35 = qword_4F8DE48[8];
      result = sub_FDC4A0(v11);
      v13 = a5;
      LOBYTE(v43[0]) = 0;
      v41 = (unsigned __int8 *)v43;
      v14 = result;
      v42 = 0;
      if ( !result )
        goto LABEL_11;
    }
    else
    {
      v11 = **(__int64 ***)(a1 + 8);
      v35 = qword_4F8DE48[8];
      v12 = sub_FDC4A0(v11);
      v13 = a5;
      LOBYTE(v43[0]) = 0;
      v41 = (unsigned __int8 *)v43;
      v14 = v12;
      v42 = 0;
      v38 = -1;
      if ( !v12 )
        goto LABEL_17;
    }
    v34 = sub_FF0420(v14, a2, a4, v13);
    v45 = 0;
    v49 = 0x100000000LL;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    v44 = (__int64 *)&unk_49DD210;
    v50 = &v41;
    sub_CB5980((__int64)&v44, 0, 0, 0);
    v40[1] = "label=\"%.1f%%\"";
    v40[0] = &unk_49DD0B8;
    *(double *)&v40[2] = (double)v34 * 100.0 * 4.656612873077393e-10;
    sub_CB6620((__int64)&v44, (__int64)v40, v15, v16, v17, v18);
    if ( v35 )
    {
      v40[0] = sub_FDD860(v11, a2);
      v26 = sub_1098D20(v40, (unsigned int)v34);
      sub_F02DB0(&v39, v35, 0x64u);
      v40[0] = *(_QWORD *)(a1 + 32);
      if ( sub_1098D20(v40, v39) <= v26 )
      {
        v27 = (_QWORD *)v48;
        if ( (unsigned __int64)(v47 - v48) > 0xB )
        {
          *(_DWORD *)(v48 + 8) = 577004914;
          *v27 = 0x223D726F6C6F632CLL;
          v19 = v48 + 12;
          v48 += 12;
          goto LABEL_8;
        }
        sub_CB6200((__int64)&v44, ",color=\"red\"", 0xCu);
      }
    }
    v19 = v48;
LABEL_8:
    if ( v46 != v19 )
      sub_CB5AE0((__int64 *)&v44);
    v44 = (__int64 *)&unk_49DD210;
    result = (__int64)sub_CB5840((__int64)&v44);
LABEL_11:
    if ( v38 > 64 )
    {
LABEL_12:
      if ( v41 != (unsigned __int8 *)v43 )
        return j_j___libc_free_0(v41, v43[0] + 1LL);
      return result;
    }
LABEL_17:
    v20 = *(_QWORD *)a1;
    v21 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v21) <= 4 )
    {
      v20 = sub_CB6200(v20, "\tNode", 5u);
    }
    else
    {
      *(_DWORD *)v21 = 1685016073;
      *(_BYTE *)(v21 + 4) = 101;
      *(_QWORD *)(v20 + 32) += 5LL;
    }
    sub_CB5A80(v20, a2);
    if ( v38 >= 0 )
    {
      v28 = *(_QWORD *)a1;
      v29 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
      if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v29 <= 1u )
      {
        v28 = sub_CB6200(v28, ":s", 2u);
      }
      else
      {
        *v29 = 29498;
        *(_QWORD *)(v28 + 32) += 2LL;
      }
      sub_CB59F0(v28, v38);
    }
    v22 = *(_QWORD *)a1;
    v23 = *(_QWORD **)(*(_QWORD *)a1 + 32LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v23 <= 7u )
    {
      v22 = sub_CB6200(v22, " -> Node", 8u);
    }
    else
    {
      *v23 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v22 + 32) += 8LL;
    }
    sub_CB5A80(v22, v36);
    if ( v42 )
    {
      v30 = *(_QWORD *)a1;
      v31 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v31 )
      {
        v30 = sub_CB6200(v30, (unsigned __int8 *)"[", 1u);
      }
      else
      {
        *v31 = 91;
        ++*(_QWORD *)(v30 + 32);
      }
      v32 = sub_CB6200(v30, v41, v42);
      v33 = *(_BYTE **)(v32 + 32);
      if ( *(_BYTE **)(v32 + 24) == v33 )
      {
        sub_CB6200(v32, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v33 = 93;
        ++*(_QWORD *)(v32 + 32);
      }
    }
    v24 = *(_QWORD *)a1;
    v25 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v25 <= 1u )
    {
      result = sub_CB6200(v24, (unsigned __int8 *)";\n", 2u);
    }
    else
    {
      result = 2619;
      *v25 = 2619;
      *(_QWORD *)(v24 + 32) += 2LL;
    }
    goto LABEL_12;
  }
  return result;
}
