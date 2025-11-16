// Function: sub_17CFB40
// Address: 0x17cfb40
//
__int64 __fastcall sub_17CFB40(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, unsigned int a5)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  _QWORD *v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // rdi
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rdx
  unsigned __int8 *v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v39; // [rsp+18h] [rbp-88h]
  __int64 *v40; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v41; // [rsp+28h] [rbp-78h] BYREF
  __int64 v42; // [rsp+30h] [rbp-70h] BYREF
  __int16 v43; // [rsp+40h] [rbp-60h]
  _BYTE v44[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v45; // [rsp+60h] [rbp-40h]

  v9 = *(_QWORD *)(a1 + 8);
  v43 = 257;
  v10 = *(_QWORD *)(v9 + 176);
  if ( v10 != *(_QWORD *)a2 )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      v45 = 257;
      v27 = sub_15FDFF0(a2, v10, (__int64)v44, 0);
      v28 = a3[1];
      a2 = v27;
      if ( v28 )
      {
        v40 = (__int64 *)a3[2];
        sub_157E9D0(v28 + 40, v27);
        v29 = *v40;
        v30 = *(_QWORD *)(a2 + 24) & 7LL;
        *(_QWORD *)(a2 + 32) = v40;
        v29 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a2 + 24) = v29 | v30;
        *(_QWORD *)(v29 + 8) = a2 + 24;
        *v40 = *v40 & 7 | (a2 + 24);
      }
      sub_164B780(a2, &v42);
      v31 = *a3;
      if ( *a3 )
      {
        v41 = (unsigned __int8 *)*a3;
        sub_1623A60((__int64)&v41, v31, 2);
        v32 = *(_QWORD *)(a2 + 48);
        v33 = a2 + 48;
        if ( v32 )
        {
          sub_161E7C0(a2 + 48, v32);
          v33 = a2 + 48;
        }
        v34 = v41;
        *(_QWORD *)(a2 + 48) = v41;
        if ( v34 )
          sub_1623210((__int64)&v41, v34, v33);
      }
      v9 = *(_QWORD *)(a1 + 8);
    }
    else
    {
      a2 = sub_15A4A70((__int64 ***)a2, v10);
      v9 = *(_QWORD *)(a1 + 8);
    }
  }
  v11 = *(_QWORD **)(v9 + 376);
  if ( *v11 )
  {
    v25 = ~*v11;
    v45 = 257;
    v26 = sub_15A0680(*(_QWORD *)(v9 + 176), v25, 0);
    a2 = sub_1281C00(a3, a2, v26, (__int64)v44);
    v9 = *(_QWORD *)(a1 + 8);
    v11 = *(_QWORD **)(v9 + 376);
  }
  v12 = v11[1];
  if ( v12 )
  {
    v45 = 257;
    v24 = sub_15A0680(*(_QWORD *)(v9 + 176), v12, 0);
    a2 = (__int64)sub_156D4C0(a3, a2, v24, (__int64)v44);
    v9 = *(_QWORD *)(a1 + 8);
    v11 = *(_QWORD **)(v9 + 376);
  }
  v13 = v11[2];
  v14 = a2;
  if ( v13 )
  {
    v45 = 257;
    v23 = sub_15A0680(*(_QWORD *)(v9 + 176), v13, 0);
    v14 = sub_12899C0(a3, a2, v23, (__int64)v44, 0, 0);
  }
  v39 = v14;
  v45 = 257;
  v15 = sub_1646BA0(a4, 0);
  v16 = sub_12AA3B0(a3, 0x2Eu, v39, v15, (__int64)v44);
  v17 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(v17 + 156) )
  {
    v18 = *(_QWORD *)(*(_QWORD *)(v17 + 376) + 24LL);
    if ( v18 )
    {
      v45 = 257;
      v37 = sub_15A0680(*(_QWORD *)(v17 + 176), v18, 0);
      a2 = sub_12899C0(a3, a2, v37, (__int64)v44, 0, 0);
      if ( a5 > 3 )
        goto LABEL_13;
    }
    else if ( a5 > 3 )
    {
LABEL_13:
      v19 = (_QWORD *)a3[3];
      v45 = 257;
      v20 = (__int64 *)sub_1643350(v19);
      v21 = sub_1646BA0(v20, 0);
      sub_12AA3B0(a3, 0x2Eu, a2, v21, (__int64)v44);
      return v16;
    }
    v35 = *(_QWORD *)(a1 + 8);
    v45 = 257;
    v36 = sub_15A0680(*(_QWORD *)(v35 + 176), -4, 0);
    a2 = sub_1281C00(a3, a2, v36, (__int64)v44);
    goto LABEL_13;
  }
  return v16;
}
