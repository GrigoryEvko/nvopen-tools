// Function: sub_1C51F30
// Address: 0x1c51f30
//
void __fastcall sub_1C51F30(__int64 ***a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int8 *v10; // rsi
  __int64 v11; // rax
  __int64 **v12; // rdx
  __int64 v13; // rax
  unsigned __int8 *v14; // rsi
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  unsigned __int8 *v22; // [rsp+8h] [rbp-C8h] BYREF
  const char *v23; // [rsp+10h] [rbp-C0h] BYREF
  char v24; // [rsp+20h] [rbp-B0h]
  char v25; // [rsp+21h] [rbp-AFh]
  unsigned __int8 *v26[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v27; // [rsp+40h] [rbp-90h]
  unsigned __int8 *v28; // [rsp+50h] [rbp-80h] BYREF
  __int64 v29; // [rsp+58h] [rbp-78h]
  __int64 *v30; // [rsp+60h] [rbp-70h]
  __int64 v31; // [rsp+68h] [rbp-68h]
  __int64 v32; // [rsp+70h] [rbp-60h]
  int v33; // [rsp+78h] [rbp-58h]
  __int64 v34; // [rsp+80h] [rbp-50h]
  __int64 v35; // [rsp+88h] [rbp-48h]

  v5 = a2;
  if ( *a1 == *(__int64 ***)a2 )
    goto LABEL_15;
  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 <= 0x17u )
  {
    v8 = a3;
  }
  else
  {
    v7 = *(_QWORD *)(a2 + 40);
    if ( v6 == 77 )
    {
      v8 = sub_157ED20(v7);
    }
    else
    {
      v15 = *(_QWORD *)(a2 + 32);
      if ( !v15 || v15 == v7 + 40 )
      {
        v28 = 0;
        v30 = 0;
        v31 = sub_16498A0(0);
        v32 = 0;
        v33 = 0;
        v34 = 0;
        v35 = 0;
        v29 = 0;
        BUG();
      }
      v8 = v15 - 24;
    }
  }
  v9 = sub_16498A0(v8);
  v34 = 0;
  v35 = 0;
  v10 = *(unsigned __int8 **)(v8 + 48);
  v31 = v9;
  v33 = 0;
  v11 = *(_QWORD *)(v8 + 40);
  v28 = 0;
  v29 = v11;
  v32 = 0;
  v30 = (__int64 *)(v8 + 24);
  v26[0] = v10;
  if ( v10 )
  {
    sub_1623A60((__int64)v26, (__int64)v10, 2);
    if ( v28 )
      sub_161E7C0((__int64)&v28, (__int64)v28);
    v28 = v26[0];
    if ( v26[0] )
      sub_1623210((__int64)v26, v26[0], (__int64)&v28);
  }
  v25 = 1;
  v12 = *a1;
  v23 = "bitCast";
  v24 = 3;
  if ( v12 == *(__int64 ***)v5 )
    goto LABEL_27;
  if ( *(_BYTE *)(v5 + 16) <= 0x10u )
  {
    v13 = sub_15A46C0(47, (__int64 ***)v5, v12, 0);
    v14 = v28;
    v5 = v13;
    goto LABEL_13;
  }
  v27 = 257;
  v16 = sub_15FDBD0(47, v5, (__int64)v12, (__int64)v26, 0);
  v5 = v16;
  if ( v29 )
  {
    v17 = v30;
    sub_157E9D0(v29 + 40, v16);
    v18 = *(_QWORD *)(v5 + 24);
    v19 = *v17;
    *(_QWORD *)(v5 + 32) = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v5 + 24) = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v5 + 24;
    *v17 = *v17 & 7 | (v5 + 24);
  }
  sub_164B780(v5, (__int64 *)&v23);
  if ( v28 )
  {
    v22 = v28;
    sub_1623A60((__int64)&v22, (__int64)v28, 2);
    v20 = *(_QWORD *)(v5 + 48);
    if ( v20 )
      sub_161E7C0(v5 + 48, v20);
    v21 = v22;
    *(_QWORD *)(v5 + 48) = v22;
    if ( v21 )
      sub_1623210((__int64)&v22, v21, v5 + 48);
LABEL_27:
    v14 = v28;
LABEL_13:
    if ( v14 )
      sub_161E7C0((__int64)&v28, (__int64)v14);
  }
LABEL_15:
  sub_1648780(a3, (__int64)a1, v5);
}
