// Function: sub_30B4C10
// Address: 0x30b4c10
//
void __fastcall sub_30B4C10(_QWORD **a1, unsigned __int64 a2, int a3, __int64 *a4, __int64 (__fastcall *a5)(__int64))
{
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  size_t v11; // rdx
  _BYTE *v12; // rsi
  __int64 v13; // r9
  int v14; // r12d
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rdi
  _QWORD *v18; // rdx
  __int64 v19; // rdi
  _WORD *v20; // rdx
  __int64 v21; // rdi
  _WORD *v22; // rdx
  __int64 v23; // rdi
  _BYTE *v24; // rax
  __int64 v25; // rdi
  _BYTE *v26; // rax
  size_t v27; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v29; // [rsp+20h] [rbp-50h] BYREF
  size_t v30; // [rsp+28h] [rbp-48h]
  _QWORD v31[8]; // [rsp+30h] [rbp-40h] BYREF

  v9 = a5(*a4);
  if ( v9 )
  {
    v10 = v9;
    v29 = (unsigned __int8 *)v31;
    sub_30B30D0((__int64 *)&v29, byte_3F871B3, (__int64)byte_3F871B3);
    v11 = v30;
    if ( v29 != (unsigned __int8 *)v31 )
    {
      v27 = v30;
      j_j___libc_free_0((unsigned __int64)v29);
      v11 = v27;
    }
    v12 = (char *)a1 + 17;
    v13 = *a1[1];
    if ( v11 )
    {
      sub_30B42A0((__int64 *)&v29, v12, a2, a4, (__int64)a5, v13);
      v14 = a3;
      if ( a3 > 64 )
      {
LABEL_6:
        if ( v29 != (unsigned __int8 *)v31 )
          j_j___libc_free_0((unsigned __int64)v29);
        return;
      }
    }
    else
    {
      sub_30B42A0((__int64 *)&v29, v12, a2, a4, (__int64)a5, v13);
      v14 = -1;
    }
    v15 = (__int64)*a1;
    v16 = (*a1)[4];
    if ( (unsigned __int64)((*a1)[3] - v16) <= 4 )
    {
      v15 = sub_CB6200(v15, "\tNode", 5u);
    }
    else
    {
      *(_DWORD *)v16 = 1685016073;
      *(_BYTE *)(v16 + 4) = 101;
      *(_QWORD *)(v15 + 32) += 5LL;
    }
    sub_CB5A80(v15, a2);
    if ( v14 >= 0 )
    {
      v21 = (__int64)*a1;
      v22 = (_WORD *)(*a1)[4];
      if ( (*a1)[3] - (_QWORD)v22 <= 1u )
      {
        v21 = sub_CB6200(v21, ":s", 2u);
      }
      else
      {
        *v22 = 29498;
        *(_QWORD *)(v21 + 32) += 2LL;
      }
      sub_CB59F0(v21, v14);
    }
    v17 = (__int64)*a1;
    v18 = (_QWORD *)(*a1)[4];
    if ( (*a1)[3] - (_QWORD)v18 <= 7u )
    {
      v17 = sub_CB6200(v17, " -> Node", 8u);
    }
    else
    {
      *v18 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v17 + 32) += 8LL;
    }
    sub_CB5A80(v17, v10);
    if ( v30 )
    {
      v23 = (__int64)*a1;
      v24 = (_BYTE *)(*a1)[4];
      if ( (_BYTE *)(*a1)[3] == v24 )
      {
        v23 = sub_CB6200(v23, (unsigned __int8 *)"[", 1u);
      }
      else
      {
        *v24 = 91;
        ++*(_QWORD *)(v23 + 32);
      }
      v25 = sub_CB6200(v23, v29, v30);
      v26 = *(_BYTE **)(v25 + 32);
      if ( *(_BYTE **)(v25 + 24) == v26 )
      {
        sub_CB6200(v25, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v26 = 93;
        ++*(_QWORD *)(v25 + 32);
      }
    }
    v19 = (__int64)*a1;
    v20 = (_WORD *)(*a1)[4];
    if ( (*a1)[3] - (_QWORD)v20 <= 1u )
    {
      sub_CB6200(v19, (unsigned __int8 *)";\n", 2u);
    }
    else
    {
      *v20 = 2619;
      *(_QWORD *)(v19 + 32) += 2LL;
    }
    goto LABEL_6;
  }
}
