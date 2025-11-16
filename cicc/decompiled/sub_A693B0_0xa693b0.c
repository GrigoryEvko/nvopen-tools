// Function: sub_A693B0
// Address: 0xa693b0
//
__int64 __fastcall sub_A693B0(__int64 a1, _BYTE *a2, __int64 a3, char a4)
{
  const __m128i *v5; // rax
  const __m128i *v6; // rdx
  unsigned __int8 v7; // al
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  unsigned __int8 v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  _BYTE *v22; // [rsp-10h] [rbp-520h]
  _QWORD v24[4]; // [rsp+10h] [rbp-500h] BYREF
  _BYTE v25[112]; // [rsp+30h] [rbp-4E0h] BYREF
  _BYTE v26[400]; // [rsp+A0h] [rbp-470h] BYREF
  __int64 v27[2]; // [rsp+230h] [rbp-2E0h] BYREF
  __int64 v28; // [rsp+240h] [rbp-2D0h]
  __int64 v29; // [rsp+248h] [rbp-2C8h]
  __int64 v30; // [rsp+250h] [rbp-2C0h]
  __int64 v31; // [rsp+258h] [rbp-2B8h]
  __int64 v32; // [rsp+260h] [rbp-2B0h]
  __int64 v33; // [rsp+268h] [rbp-2A8h]
  __int64 v34; // [rsp+270h] [rbp-2A0h]
  __int64 v35; // [rsp+278h] [rbp-298h]
  __int64 v36; // [rsp+280h] [rbp-290h]
  __int64 v37; // [rsp+288h] [rbp-288h]
  __int64 v38; // [rsp+290h] [rbp-280h]
  __int64 v39; // [rsp+298h] [rbp-278h]
  __int64 v40; // [rsp+2A0h] [rbp-270h]
  __int64 v41; // [rsp+2A8h] [rbp-268h]
  __int64 v42; // [rsp+2B0h] [rbp-260h]
  __int64 v43; // [rsp+2B8h] [rbp-258h]
  __int64 v44; // [rsp+2C0h] [rbp-250h]
  __int64 v45; // [rsp+2C8h] [rbp-248h]
  __int64 v46; // [rsp+2D0h] [rbp-240h]
  __int64 v47; // [rsp+2D8h] [rbp-238h]
  __int64 v48; // [rsp+2E0h] [rbp-230h]
  __int64 v49; // [rsp+2E8h] [rbp-228h]
  __int64 v50; // [rsp+2F0h] [rbp-220h]
  __int64 v51; // [rsp+2F8h] [rbp-218h]
  __int64 v52; // [rsp+300h] [rbp-210h]
  __int64 v53; // [rsp+308h] [rbp-208h]

  sub_A54BD0((__int64)v25, (__int64)a2);
  sub_A55A10((__int64)v26, 0, 0);
  v5 = sub_A56340(a3, 0);
  v6 = (const __m128i *)v26;
  if ( v5 )
    v6 = sub_A56340(a3, 0);
  v7 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    v8 = *(_QWORD *)(a1 + 40);
    if ( v8 )
    {
      v9 = *(_QWORD *)(v8 + 72);
      if ( v9 )
        sub_A564B0(a3, v9);
    }
    v10 = sub_A4F760((unsigned __int8 *)a1);
    sub_A685A0((__int64)v27, (__int64)v25, v11, v10, v12, a4, 0);
    a2 = (_BYTE *)a1;
    sub_A635F0(v27, (unsigned __int8 *)a1);
    sub_A555E0((__int64)v27);
    goto LABEL_8;
  }
  if ( v7 == 23 )
  {
    v14 = *(_QWORD *)(a1 + 72);
    if ( v14 )
      sub_A564B0(a3, v14);
    v15 = sub_A4F760((unsigned __int8 *)a1);
    sub_A685A0((__int64)v27, (__int64)v25, v16, v15, v17, a4, 0);
    sub_A651F0(v27, a1);
    sub_A555E0((__int64)v27);
    a2 = v22;
    goto LABEL_8;
  }
  if ( v7 <= 3u )
  {
    sub_A685A0((__int64)v27, (__int64)v25, (__int64)v6, *(_QWORD *)(a1 + 40), 0, a4, 0);
    v18 = *(_BYTE *)a1;
    if ( *(_BYTE *)a1 == 3 )
    {
      a2 = (_BYTE *)a1;
      sub_A62020(v27, a1);
LABEL_16:
      sub_A555E0((__int64)v27);
      goto LABEL_8;
    }
    switch ( v18 )
    {
      case 0u:
        a2 = (_BYTE *)a1;
        sub_A65640(v27, a1);
        goto LABEL_16;
      case 1u:
        a2 = (_BYTE *)a1;
        sub_A62650(v27, a1);
        goto LABEL_16;
      case 2u:
        a2 = (_BYTE *)a1;
        sub_A62950(v27, a1);
        goto LABEL_16;
    }
LABEL_34:
    BUG();
  }
  if ( v7 == 24 )
  {
    v19 = sub_A4F760((unsigned __int8 *)a1);
    sub_A62C00(*(const char **)(a1 + 24), (__int64)a2, v20, v19);
    goto LABEL_8;
  }
  if ( v7 > 0x15u )
  {
    if ( v7 == 25 || v7 == 22 )
    {
      a2 = v25;
      sub_A5C020((_BYTE *)a1, (__int64)v25, 1, a3);
      goto LABEL_8;
    }
    goto LABEL_34;
  }
  v21 = *(_QWORD *)(a1 + 8);
  v27[0] = 0;
  v27[1] = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  sub_A57EC0((__int64)v27, v21, (__int64)v25);
  sub_A51310((__int64)v25, 0x20u);
  v24[0] = off_4979428;
  v24[1] = v27;
  v24[2] = sub_A56340(a3, 32);
  v24[3] = 0;
  sub_A5AA10((__int64)v25, (char *)a1, (__int64)v24);
  if ( v51 )
    j_j___libc_free_0(v51, v53 - v51);
  sub_C7D6A0(v48, 16LL * (unsigned int)v50, 8);
  if ( v43 )
    j_j___libc_free_0(v43, v45 - v43);
  sub_C7D6A0(v40, 8LL * (unsigned int)v42, 8);
  sub_C7D6A0(v36, 8LL * (unsigned int)v38, 8);
  sub_C7D6A0(v32, 8LL * (unsigned int)v34, 8);
  a2 = (_BYTE *)(8LL * (unsigned int)v30);
  sub_C7D6A0(v28, a2, 8);
LABEL_8:
  sub_A552A0((__int64)v26, (__int64)a2);
  return sub_A54D10((__int64)v25, (__int64)a2);
}
