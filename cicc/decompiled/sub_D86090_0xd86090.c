// Function: sub_D86090
// Address: 0xd86090
//
__int64 __fastcall sub_D86090(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  __int64 v6; // r12
  unsigned __int64 v7; // r14
  _QWORD *v8; // rcx
  unsigned __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  bool v13; // al
  _BOOL8 v14; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  _QWORD *v18; // [rsp+8h] [rbp-38h]
  unsigned __int64 v19; // [rsp+8h] [rbp-38h]
  _QWORD *v20; // [rsp+8h] [rbp-38h]

  v6 = sub_22077B0(144);
  v7 = **a3;
  *(_QWORD *)(v6 + 32) = v7;
  memset((void *)(v6 + 40), 0, 0x68u);
  v8 = a1 + 1;
  *(_QWORD *)(v6 + 64) = v6 + 48;
  *(_QWORD *)(v6 + 72) = v6 + 48;
  *(_QWORD *)(v6 + 112) = v6 + 96;
  *(_QWORD *)(v6 + 120) = v6 + 96;
  if ( a1 + 1 == a2 )
  {
    if ( a1[5] )
    {
      v11 = a1[4];
      if ( *(_QWORD *)(v11 + 32) < v7 )
        goto LABEL_19;
    }
  }
  else
  {
    v9 = a2[4];
    if ( v7 < v9 )
    {
      if ( (_QWORD *)a1[3] == a2 )
        goto LABEL_21;
      v10 = sub_220EF80(a2);
      v8 = a1 + 1;
      v11 = v10;
      if ( *(_QWORD *)(v10 + 32) >= v7 )
        goto LABEL_5;
      if ( *(_QWORD *)(v10 + 24) )
      {
LABEL_21:
        v12 = a2;
LABEL_22:
        v11 = (__int64)a2;
        goto LABEL_6;
      }
LABEL_19:
      v13 = 0;
LABEL_7:
      if ( v8 == (_QWORD *)v11 || v13 )
        goto LABEL_9;
      v17 = *(_QWORD *)(v11 + 32);
      goto LABEL_24;
    }
    v19 = a2[4];
    v12 = a2;
    if ( v7 <= v9 )
      goto LABEL_16;
    if ( (_QWORD *)a1[4] == a2 )
    {
      v12 = 0;
      goto LABEL_22;
    }
    v16 = sub_220EEE0(a2);
    v8 = a1 + 1;
    v11 = v16;
    if ( *(_QWORD *)(v16 + 32) > v7 )
    {
      v17 = v19;
      if ( a2[3] )
      {
LABEL_9:
        v14 = 1;
LABEL_10:
        sub_220F040(v14, v6, v11, v8);
        ++a1[5];
        return v6;
      }
      v11 = (__int64)a2;
LABEL_24:
      v14 = v17 > v7;
      goto LABEL_10;
    }
  }
LABEL_5:
  v18 = v8;
  v12 = sub_D85400((__int64)a1, (unsigned __int64 *)(v6 + 32));
  v8 = v18;
  if ( v11 )
  {
LABEL_6:
    v13 = v12 != 0;
    goto LABEL_7;
  }
LABEL_16:
  v20 = v12;
  sub_D85F30(0);
  sub_D85E30(*(_QWORD **)(v6 + 56));
  j_j___libc_free_0(v6, 144);
  return (__int64)v20;
}
