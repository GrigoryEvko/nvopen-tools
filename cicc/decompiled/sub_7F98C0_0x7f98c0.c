// Function: sub_7F98C0
// Address: 0x7f98c0
//
void __fastcall sub_7F98C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v9; // rdi
  __int64 v11; // r10
  __int64 v12; // rbx
  __int64 v13; // r13
  _QWORD *v14; // rax
  _BYTE *v15; // r15
  int v16; // eax
  __int64 v17; // r10
  __int64 *v18; // r8
  __int64 v19; // rax
  __m128i *v20; // rax
  __int64 *v21; // rax
  __int64 v22; // r13
  __int64 *v23; // rax
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdi
  __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rsi
  _BYTE *v36; // rax
  __int64 v37; // [rsp+8h] [rbp-58h]
  _QWORD *v38; // [rsp+8h] [rbp-58h]
  __int64 v39; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  __int64 *v41; // [rsp+10h] [rbp-50h]
  __int64 *v42; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+18h] [rbp-48h]
  __int64 v45; // [rsp+18h] [rbp-48h]
  char v46[4]; // [rsp+24h] [rbp-3Ch] BYREF
  _QWORD v47[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = *(_QWORD *)(a1 + 40);
  v47[0] = 0;
  if ( !v7 )
  {
    if ( a5 )
      sub_7E25D0(a5, (int *)a6);
    return;
  }
  v9 = *(_QWORD *)(v7 + 16);
  if ( v9 )
  {
    v39 = a5;
    sub_87ADD0(v9, v46, v47, (char *)v47 + 4);
    v11 = *(_QWORD *)(a6 + 8);
    a5 = v39;
    if ( (*(_BYTE *)a1 & 2) == 0 && !v47[0] )
    {
LABEL_7:
      if ( *(char *)(v7 + 49) >= 0 )
      {
        v12 = *(_QWORD *)(v7 + 80);
        if ( a5 )
          sub_7E25D0(a5, (int *)a6);
        v13 = *(_QWORD *)(v12 + 88);
        if ( v13 )
        {
          v14 = sub_73A830(0, 5u);
          sub_7E6AB0(v13, (__int64)v14, (int *)a6);
        }
        return;
      }
    }
  }
  else
  {
    v11 = *(_QWORD *)(a6 + 8);
    if ( (*(_BYTE *)a1 & 2) == 0 )
      goto LABEL_7;
  }
  if ( !(v11 | a5) )
    return;
  v37 = a5;
  v40 = v11;
  v15 = sub_7F98A0(a2, 0);
  v16 = sub_8D3410(*(_QWORD *)(a1 + 8));
  v17 = v40;
  v18 = (__int64 *)v37;
  if ( v16 )
  {
    v24 = sub_691620(*(_QWORD *)(a1 + 8));
    v25 = sub_691630(v24, 1);
    v17 = v40;
    v18 = (__int64 *)v37;
    if ( v25 )
    {
      v26 = sub_8D4050(*(_QWORD *)(a1 + 8));
      v27 = sub_7F5F50(v26, a4);
      v17 = v40;
      v18 = (__int64 *)v37;
      v28 = v27;
      if ( v27 )
      {
        v42 = (__int64 *)v37;
        v45 = v17;
        v29 = byte_4F06A51[0];
        v30 = sub_73A830(v27, byte_4F06A51[0]);
        v17 = v45;
        v18 = (__int64 *)v37;
        v38 = v30;
        if ( v30 )
        {
          v33 = sub_7E1C30(v28, v29, v31, v32, v18);
          v34 = (__int64 *)sub_73E130(v15, v33);
          v35 = *v34;
          v34[2] = (__int64)v38;
          v36 = sub_73DBF0(0x33u, v35, (__int64)v34);
          v18 = v42;
          v17 = v45;
          v15 = v36;
        }
      }
    }
  }
  v41 = v18;
  v44 = v17;
  v19 = sub_7E1C10();
  v20 = (__m128i *)sub_73E130(v15, v19);
  v20[1].m128i_i64[0] = a3;
  sub_7F88E0(*(_QWORD *)(v7 + 16), v20);
  if ( (*(_BYTE *)a1 & 2) != 0 || v47[0] )
  {
    v23 = sub_73DFE0(v44, v41);
    v22 = sub_7DEB30((__int64)v23);
  }
  else
  {
    if ( !v41 )
      return;
    v21 = (__int64 *)sub_7DEB30((__int64)v41);
    v22 = (__int64)sub_73DFE0(v44, v21);
  }
  if ( v22 )
  {
    sub_7E1790(a6);
    sub_7E25D0(v22, (int *)a6);
  }
}
