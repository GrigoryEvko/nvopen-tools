// Function: sub_139A3C0
// Address: 0x139a3c0
//
void __fastcall sub_139A3C0(__int64 *a1, __int64 a2, int a3, _QWORD *a4, __int64 (__fastcall *a5)(__int64 *))
{
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  int v11; // r12d
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdi
  _QWORD *v15; // rdx
  __int64 v16; // rdi
  _WORD *v17; // rdx
  __int64 v18; // rdi
  _WORD *v19; // rdx
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 v24; // [rsp+0h] [rbp-60h]
  __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  __int64 v27; // [rsp+18h] [rbp-48h]
  _QWORD v28[8]; // [rsp+20h] [rbp-40h] BYREF

  v8 = a4[2];
  v26 = 6;
  v27 = 0;
  v28[0] = v8;
  if ( v8 != 0 && v8 != -8 && v8 != -16 )
    sub_1649AC0(&v26, *a4 & 0xFFFFFFFFFFFFFFF8LL);
  v28[1] = a4[3];
  v9 = a5(&v26);
  if ( v28[0] != 0 && v28[0] != -8 && v28[0] != -16 )
    sub_1649B30(&v26);
  if ( v9 )
  {
    v26 = (__int64)v28;
    sub_1399600(&v26, byte_3F871B3, (__int64)byte_3F871B3);
    v10 = v27;
    if ( (_QWORD *)v26 != v28 )
    {
      v24 = v27;
      j_j___libc_free_0(v26, v28[0] + 1LL);
      v10 = v24;
    }
    v26 = (__int64)v28;
    if ( v10 )
    {
      sub_1399600(&v26, byte_3F871B3, (__int64)byte_3F871B3);
      v11 = a3;
      if ( a3 > 64 )
      {
LABEL_12:
        if ( (_QWORD *)v26 != v28 )
          j_j___libc_free_0(v26, v28[0] + 1LL);
        return;
      }
    }
    else
    {
      sub_1399600(&v26, byte_3F871B3, (__int64)byte_3F871B3);
      v11 = -1;
    }
    v12 = *a1;
    v13 = *(_QWORD *)(*a1 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v13) <= 4 )
    {
      v12 = sub_16E7EE0(v12, "\tNode", 5);
    }
    else
    {
      *(_DWORD *)v13 = 1685016073;
      *(_BYTE *)(v13 + 4) = 101;
      *(_QWORD *)(v12 + 24) += 5LL;
    }
    sub_16E7B40(v12, a2);
    if ( v11 >= 0 )
    {
      v18 = *a1;
      v19 = *(_WORD **)(*a1 + 24);
      if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v19 <= 1u )
      {
        v18 = sub_16E7EE0(v18, ":s", 2);
      }
      else
      {
        *v19 = 29498;
        *(_QWORD *)(v18 + 24) += 2LL;
      }
      sub_16E7AB0(v18, v11);
    }
    v14 = *a1;
    v15 = *(_QWORD **)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v15 <= 7u )
    {
      v14 = sub_16E7EE0(v14, " -> Node", 8);
    }
    else
    {
      *v15 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v14 + 24) += 8LL;
    }
    sub_16E7B40(v14, v9);
    if ( v27 )
    {
      v20 = *a1;
      v21 = *(_BYTE **)(*a1 + 24);
      if ( *(_BYTE **)(*a1 + 16) == v21 )
      {
        v20 = sub_16E7EE0(v20, "[", 1);
      }
      else
      {
        *v21 = 91;
        ++*(_QWORD *)(v20 + 24);
      }
      v22 = sub_16E7EE0(v20, (const char *)v26, v27);
      v23 = *(_BYTE **)(v22 + 24);
      if ( *(_BYTE **)(v22 + 16) == v23 )
      {
        sub_16E7EE0(v22, "]", 1);
      }
      else
      {
        *v23 = 93;
        ++*(_QWORD *)(v22 + 24);
      }
    }
    v16 = *a1;
    v17 = *(_WORD **)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v17 <= 1u )
    {
      sub_16E7EE0(v16, ";\n", 2);
    }
    else
    {
      *v17 = 2619;
      *(_QWORD *)(v16 + 24) += 2LL;
    }
    goto LABEL_12;
  }
}
