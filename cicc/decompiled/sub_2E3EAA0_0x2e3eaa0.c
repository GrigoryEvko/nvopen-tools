// Function: sub_2E3EAA0
// Address: 0x2e3eaa0
//
void __fastcall sub_2E3EAA0(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  unsigned __int64 v4; // r13
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdi
  _QWORD *v15; // rdx
  __int64 v16; // rdi
  _WORD *v17; // rdx
  unsigned __int64 v18; // r15
  _QWORD *v19; // rdx
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rdi
  _BYTE *v23; // rax
  signed int v24; // [rsp+10h] [rbp-D0h]
  unsigned int v25; // [rsp+14h] [rbp-CCh]
  unsigned int v26; // [rsp+2Ch] [rbp-B4h] BYREF
  unsigned __int64 v27[4]; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int8 *v28; // [rsp+50h] [rbp-90h] BYREF
  size_t v29; // [rsp+58h] [rbp-88h]
  _BYTE v30[16]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v31[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v32; // [rsp+80h] [rbp-60h]
  __int64 v33; // [rsp+88h] [rbp-58h]
  __int64 v34; // [rsp+90h] [rbp-50h]
  __int64 v35; // [rsp+98h] [rbp-48h]
  unsigned __int8 **v36; // [rsp+A0h] [rbp-40h]

  v4 = *a4;
  if ( !*a4 )
    return;
  v7 = **(__int64 ***)(a1 + 8);
  v25 = qword_4F8DE48[8];
  v8 = sub_2E3A070(v7);
  v30[0] = 0;
  v29 = 0;
  v28 = v30;
  if ( v8 )
  {
    v24 = sub_2E441C0(v8, a2, a4);
    v35 = 0x100000000LL;
    v31[1] = 0;
    v32 = 0;
    v33 = 0;
    v31[0] = (__int64)&unk_49DD210;
    v36 = &v28;
    v34 = 0;
    sub_CB5980((__int64)v31, 0, 0, 0);
    v27[1] = (unsigned __int64)"label=\"%.1f%%\"";
    v27[0] = (unsigned __int64)&unk_49DD0B8;
    *(double *)&v27[2] = (double)v24 * 100.0 * 4.656612873077393e-10;
    sub_CB6620((__int64)v31, (__int64)v27, (__int64)"label=\"%.1f%%\"", v9, v10, (__int64)v27);
    if ( v25 )
    {
      v27[0] = sub_2E39EA0(v7, a2);
      v18 = sub_1098D20(v27, v24);
      sub_F02DB0(&v26, v25, 0x64u);
      v27[0] = *(_QWORD *)(a1 + 32);
      if ( sub_1098D20(v27, v26) <= v18 )
      {
        v19 = (_QWORD *)v34;
        if ( (unsigned __int64)(v33 - v34) > 0xB )
        {
          *(_DWORD *)(v34 + 8) = 577004914;
          *v19 = 0x223D726F6C6F632CLL;
          v11 = v34 + 12;
          v34 += 12;
          goto LABEL_5;
        }
        sub_CB6200((__int64)v31, ",color=\"red\"", 0xCu);
      }
    }
    v11 = v34;
LABEL_5:
    if ( v32 != v11 )
      sub_CB5AE0(v31);
    v31[0] = (__int64)&unk_49DD210;
    sub_CB5840((__int64)v31);
  }
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v13) <= 4 )
  {
    v12 = sub_CB6200(v12, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v13 = 1685016073;
    *(_BYTE *)(v13 + 4) = 101;
    *(_QWORD *)(v12 + 32) += 5LL;
  }
  sub_CB5A80(v12, a2);
  v14 = *(_QWORD *)a1;
  v15 = *(_QWORD **)(*(_QWORD *)a1 + 32LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v15 <= 7u )
  {
    v14 = sub_CB6200(v14, " -> Node", 8u);
  }
  else
  {
    *v15 = 0x65646F4E203E2D20LL;
    *(_QWORD *)(v14 + 32) += 8LL;
  }
  sub_CB5A80(v14, v4);
  if ( v29 )
  {
    v20 = *(_QWORD *)a1;
    v21 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
    if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v21 )
    {
      v20 = sub_CB6200(v20, (unsigned __int8 *)"[", 1u);
    }
    else
    {
      *v21 = 91;
      ++*(_QWORD *)(v20 + 32);
    }
    v22 = sub_CB6200(v20, v28, v29);
    v23 = *(_BYTE **)(v22 + 32);
    if ( *(_BYTE **)(v22 + 24) == v23 )
    {
      sub_CB6200(v22, (unsigned __int8 *)"]", 1u);
    }
    else
    {
      *v23 = 93;
      ++*(_QWORD *)(v22 + 32);
    }
  }
  v16 = *(_QWORD *)a1;
  v17 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v17 <= 1u )
  {
    sub_CB6200(v16, (unsigned __int8 *)";\n", 2u);
  }
  else
  {
    *v17 = 2619;
    *(_QWORD *)(v16 + 32) += 2LL;
  }
  if ( v28 != v30 )
    j_j___libc_free_0((unsigned __int64)v28);
}
