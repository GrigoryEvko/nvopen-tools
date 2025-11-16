// Function: sub_17CE860
// Address: 0x17ce860
//
__int64 __fastcall sub_17CE860(__int64 a1, __int64 a2, __int64 *a3, int a4)
{
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // r12
  __int64 *v15; // rax
  __int64 **v16; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 *v25; // r13
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rsi
  unsigned __int8 *v30; // rsi
  __int64 *v31; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+18h] [rbp-78h] BYREF
  __int64 v34[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v35; // [rsp+30h] [rbp-60h]
  _BYTE v36[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v37; // [rsp+50h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 16);
  v35 = 257;
  v8 = *(_QWORD *)(v7 + 224);
  v9 = *(_QWORD *)(v7 + 176);
  v10 = *(_QWORD *)v8;
  if ( v9 != *(_QWORD *)v8 )
  {
    if ( *(_BYTE *)(v8 + 16) > 0x10u )
    {
      v18 = *(_QWORD *)(v7 + 224);
      v37 = 257;
      v19 = sub_15FDFF0(v18, v9, (__int64)v36, 0);
      v20 = a3[1];
      v8 = v19;
      if ( v20 )
      {
        v31 = (__int64 *)a3[2];
        sub_157E9D0(v20 + 40, v19);
        v21 = *v31;
        v22 = *(_QWORD *)(v8 + 24) & 7LL;
        *(_QWORD *)(v8 + 32) = v31;
        v21 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v21 | v22;
        *(_QWORD *)(v21 + 8) = v8 + 24;
        *v31 = *v31 & 7 | (v8 + 24);
      }
      sub_164B780(v8, v34);
      sub_12A86E0(a3, v8);
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL);
    }
    else
    {
      v8 = sub_15A4A70(*(__int64 ****)(v7 + 224), v9);
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 176LL);
    }
  }
  v37 = 257;
  v11 = sub_15A0680(v10, a4, 0);
  v12 = sub_12899C0(a3, v8, v11, (__int64)v36, 0, 0);
  v13 = *(_QWORD **)(a1 + 24);
  v14 = v12;
  v35 = 259;
  v34[0] = (__int64)"_msarg";
  v15 = sub_17CD8D0(v13, a2);
  v16 = (__int64 **)sub_1646BA0(v15, 0);
  if ( v16 != *(__int64 ***)v14 )
  {
    if ( *(_BYTE *)(v14 + 16) > 0x10u )
    {
      v37 = 257;
      v23 = sub_15FDBD0(46, v14, (__int64)v16, (__int64)v36, 0);
      v24 = a3[1];
      v14 = v23;
      if ( v24 )
      {
        v25 = (__int64 *)a3[2];
        sub_157E9D0(v24 + 40, v23);
        v26 = *(_QWORD *)(v14 + 24);
        v27 = *v25;
        *(_QWORD *)(v14 + 32) = v25;
        v27 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v14 + 24) = v27 | v26 & 7;
        *(_QWORD *)(v27 + 8) = v14 + 24;
        *v25 = *v25 & 7 | (v14 + 24);
      }
      sub_164B780(v14, v34);
      v28 = *a3;
      if ( *a3 )
      {
        v33 = *a3;
        sub_1623A60((__int64)&v33, v28, 2);
        v29 = *(_QWORD *)(v14 + 48);
        if ( v29 )
          sub_161E7C0(v14 + 48, v29);
        v30 = (unsigned __int8 *)v33;
        *(_QWORD *)(v14 + 48) = v33;
        if ( v30 )
          sub_1623210((__int64)&v33, v30, v14 + 48);
      }
    }
    else
    {
      return sub_15A46C0(46, (__int64 ***)v14, v16, 0);
    }
  }
  return v14;
}
