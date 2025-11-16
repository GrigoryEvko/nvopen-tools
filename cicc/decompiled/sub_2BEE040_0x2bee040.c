// Function: sub_2BEE040
// Address: 0x2bee040
//
__int64 __fastcall sub_2BEE040(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 *v7; // r14
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 *v12; // rax
  _QWORD *v13; // rax
  unsigned __int64 v14; // r14
  unsigned __int8 *v15; // rsi
  __int64 v16; // r13
  _BYTE *v17; // rax
  _BYTE *v18; // rdi
  __int64 v19; // r15
  __int64 *v20; // rbx
  __int64 *v21; // r12
  __int64 v22; // rdi
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rbx
  unsigned __int64 v26; // r14
  unsigned __int8 v27; // [rsp+Fh] [rbp-51h]
  _QWORD v28[10]; // [rsp+10h] [rbp-50h] BYREF

  v5 = a2;
  v6 = a1;
  if ( !*(_QWORD *)(a1 + 24) )
  {
    a1 = 528;
    v23 = sub_B2BE50(a2);
    v24 = sub_22077B0(0x210u);
    v25 = v24;
    if ( v24 )
    {
      a2 = v23;
      a1 = v24;
      sub_31867E0(v24, v23);
    }
    v26 = *(_QWORD *)(v6 + 24);
    *(_QWORD *)(v6 + 24) = v25;
    if ( v26 )
    {
      sub_3186A70(v26);
      a2 = 528;
      a1 = v26;
      j_j___libc_free_0(v26);
    }
  }
  if ( (_BYTE)qword_5010908 )
  {
    v13 = sub_CB7210(a1, a2, a3, a4, a5);
    v14 = *(_QWORD *)(v6 + 48);
    v15 = *(unsigned __int8 **)(v6 + 40);
    v16 = (__int64)v13;
    v17 = (_BYTE *)v13[3];
    v18 = *(_BYTE **)(v16 + 32);
    if ( v14 > v17 - v18 )
    {
      v19 = sub_CB6200(v16, v15, *(_QWORD *)(v6 + 48));
      v17 = *(_BYTE **)(v19 + 24);
      v18 = *(_BYTE **)(v19 + 32);
    }
    else
    {
      v19 = v16;
      if ( v14 )
      {
        memcpy(v18, v15, *(_QWORD *)(v6 + 48));
        v17 = *(_BYTE **)(v16 + 24);
        v18 = (_BYTE *)(v14 + *(_QWORD *)(v16 + 32));
        *(_QWORD *)(v16 + 32) = v18;
      }
    }
    if ( v17 == v18 )
    {
      sub_CB6200(v19, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v18 = 10;
      ++*(_QWORD *)(v19 + 32);
    }
    v20 = *(__int64 **)(v6 + 72);
    v21 = &v20[*(unsigned int *)(v6 + 80)];
    while ( v21 != v20 )
    {
      v22 = *v20++;
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v22 + 16LL))(v22, v16);
    }
    return 0;
  }
  if ( sub_2241AC0((__int64)&qword_50106C8[8], ".*")
    && !(unsigned __int8)sub_2BEDA60(v6, (unsigned __int64 *)(*(_QWORD *)(v5 + 40) + 200LL)) )
  {
    return 0;
  }
  v7 = *(__int64 **)v6;
  sub_DFB180(*(__int64 **)v6, 1u);
  if ( !(unsigned int)sub_DFB120((__int64)v7) || (unsigned __int8)sub_B2D610(v5, 30) )
    return 0;
  v9 = sub_318A600(*(_QWORD *)(v6 + 24), v5);
  v10 = *(_QWORD *)(v6 + 8);
  v11 = v9;
  v12 = *(__int64 **)v6;
  v28[1] = *(_QWORD *)(v6 + 16);
  v28[0] = v10;
  v28[2] = v12;
  v27 = sub_318D240(v6 + 32, v11, v28);
  sub_3186DF0(*(_QWORD *)(v6 + 24));
  return v27;
}
