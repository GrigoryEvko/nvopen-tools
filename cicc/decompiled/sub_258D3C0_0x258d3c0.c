// Function: sub_258D3C0
// Address: 0x258d3c0
//
__int64 __fastcall sub_258D3C0(__int64 a1, char a2)
{
  _BYTE *v3; // r9
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned __int8 v6; // al
  _BYTE *v7; // rdi
  __int64 v8; // r14
  __int64 *v9; // r12
  _QWORD *v10; // rdi
  __int64 v11; // r15
  unsigned __int64 v12; // r9
  unsigned __int8 v13; // al
  __int64 v14; // rax
  unsigned __int8 *v15; // r8
  __int64 (*v16)(void); // rax
  _BYTE *v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rdi
  _BYTE *v20; // r9
  unsigned __int8 *v21; // rcx
  __int64 v22; // r8
  unsigned __int64 v23; // rax
  unsigned __int8 v24; // dl
  unsigned __int8 v25; // di
  unsigned __int64 v26; // r13
  __int64 v28; // rax
  __m128i v29; // rax
  __int64 v30; // rsi
  unsigned __int64 v31; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v32; // [rsp+18h] [rbp-A8h]
  __int64 v33; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v34; // [rsp+22h] [rbp-9Eh]
  _BYTE *v37; // [rsp+28h] [rbp-98h]
  __int64 *v38; // [rsp+28h] [rbp-98h]
  __m128i v39; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v40; // [rsp+50h] [rbp-70h] BYREF
  __int64 v41; // [rsp+58h] [rbp-68h]
  _BYTE v42[96]; // [rsp+60h] [rbp-60h] BYREF

  v40 = v42;
  v3 = *(_BYTE **)(a1 + 24);
  v4 = *(_QWORD *)a1;
  v5 = *(_QWORD *)(a1 + 16);
  v41 = 0x300000000LL;
  v37 = v3;
  sub_250D230((unsigned __int64 *)&v39, **(_QWORD **)(a1 + 8), 2, 0);
  v6 = sub_2526B50(v4, &v39, v5, (__int64)&v40, a2, v37, 1u);
  v7 = v40;
  v34 = v6;
  if ( v6 )
  {
    v8 = 16LL * (unsigned int)v41;
    v38 = (__int64 *)&v40[v8];
    if ( &v40[v8] != v40 )
    {
      v9 = (__int64 *)v40;
      while ( 1 )
      {
        while ( 1 )
        {
          v18 = *v9;
          v19 = *(_QWORD *)a1;
          v20 = *(_BYTE **)(a1 + 24);
          v21 = **(unsigned __int8 ***)(a1 + 32);
          v22 = *(_QWORD *)(a1 + 16);
          v39.m128i_i8[8] = 1;
          v39.m128i_i64[0] = v18;
          v23 = sub_2527B10(v19, v18, v39.m128i_i64[1], v21, v22, v20);
          v25 = v24;
          v26 = v23;
          if ( v24 )
            break;
LABEL_13:
          v9 += 2;
          if ( v38 == v9 )
            goto LABEL_27;
        }
        if ( !v23 )
          break;
        if ( !(unsigned __int8)sub_252BB70(*(_QWORD *)a1, *(_QWORD *)(a1 + 16), v23, 1) )
          goto LABEL_17;
        v33 = *(_QWORD *)(a1 + 16);
        v29.m128i_i64[0] = sub_250D2C0(v26, 0);
        v30 = *(_QWORD *)a1;
        v39 = v29;
        if ( !(unsigned __int8)sub_258BE70(v33, v30, &v39, a2) )
          goto LABEL_17;
        v9 += 2;
        if ( v38 == v9 )
        {
LABEL_27:
          v7 = v40;
          goto LABEL_20;
        }
      }
      v26 = v18;
LABEL_17:
      if ( a2 == 1 && !(unsigned __int8)sub_250C180(v26, **(_QWORD **)(a1 + 40)) )
      {
        sub_258C650(*(_QWORD *)(a1 + 16), *(_QWORD *)a1);
        v34 = v25;
        v7 = v40;
        goto LABEL_20;
      }
      v10 = *(_QWORD **)(a1 + 16);
      v11 = (__int64)v10;
      v12 = v10[9] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v10[9] & 3LL) == 3 )
        v12 = *(_QWORD *)(v12 + 24);
      v13 = *(_BYTE *)v12;
      if ( *(_BYTE *)v12 )
      {
        if ( v13 == 22 )
        {
          v12 = *(_QWORD *)(v12 + 24);
        }
        else if ( v13 <= 0x1Cu )
        {
          v12 = 0;
        }
        else
        {
          v14 = sub_B43CB0(v12);
          v10 = *(_QWORD **)(a1 + 16);
          v12 = v14;
        }
      }
      v15 = **(unsigned __int8 ***)(a1 + 32);
      v16 = *(__int64 (**)(void))(*v10 + 40LL);
      if ( (char *)v16 == (char *)sub_2505DE0 )
      {
        v17 = v10 + 11;
      }
      else
      {
        v31 = v12;
        v32 = **(unsigned __int8 ***)(a1 + 32);
        v28 = v16();
        v12 = v31;
        v15 = v32;
        v17 = (_BYTE *)v28;
      }
      sub_258BA20(v11, *(_QWORD *)a1, v17, v26, v15, a2, v12);
      goto LABEL_13;
    }
  }
LABEL_20:
  if ( v7 != v42 )
    _libc_free((unsigned __int64)v7);
  return v34;
}
