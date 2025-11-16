// Function: sub_2FA7A40
// Address: 0x2fa7a40
//
__int64 __fastcall sub_2FA7A40(__int64 *a1, __int64 **a2)
{
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 *v6; // rdi
  __int64 v7; // rax
  __int64 *v8; // rsi
  __int64 *v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 *v12; // rsi
  _QWORD *v13; // r13
  __int64 *v14; // r8
  __int64 *v15; // rsi
  __int64 *v16; // rsi
  _BYTE *v17; // rax
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // r15
  _QWORD *v25; // rbx
  __int64 v26; // [rsp+18h] [rbp-88h]
  __int64 *v27; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v28; // [rsp+28h] [rbp-78h]
  __int64 *v29; // [rsp+30h] [rbp-70h]
  _QWORD v30[4]; // [rsp+40h] [rbp-60h] BYREF
  char v31; // [rsp+60h] [rbp-40h]
  char v32; // [rsp+61h] [rbp-3Fh]

  v4 = a2[4];
  if ( v4 == (__int64 *)(a2 + 3) )
    return 0;
  while ( 1 )
  {
    if ( !v4 )
      BUG();
    if ( (*((_BYTE *)v4 - 53) & 0x40) != 0 )
    {
      v5 = sub_B2DBE0((__int64)(v4 - 7));
      if ( !sub_2241AC0(v5, "shadow-stack") )
        break;
    }
    v4 = (__int64 *)v4[1];
    if ( a2 + 3 == (__int64 **)v4 )
      return 0;
  }
  v6 = *a2;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30[0] = sub_BCB2D0(v6);
  sub_9183A0((__int64)&v27, 0, v30);
  v7 = sub_BCB2D0(*a2);
  v8 = v28;
  v30[0] = v7;
  if ( v28 == v29 )
  {
    sub_9183A0((__int64)&v27, v28, v30);
    v9 = v28;
  }
  else
  {
    if ( v28 )
    {
      *v28 = v7;
      v8 = v28;
    }
    v9 = v8 + 1;
    v28 = v9;
  }
  a1[2] = sub_BD0EC0(v27, v9 - v27, "gc_map", 6u, 0);
  v10 = sub_BCE3C0(*a2, 0);
  v11 = sub_BCE3C0(*a2, 0);
  v12 = v27;
  v13 = (_QWORD *)v11;
  if ( v27 != v28 )
    v28 = v27;
  v14 = v29;
  v30[0] = v11;
  if ( v29 == v27 )
  {
    sub_9183A0((__int64)&v27, v27, v30);
    v15 = v28;
    v30[0] = v10;
    if ( v29 != v28 )
    {
      if ( !v28 )
        goto LABEL_18;
      goto LABEL_17;
    }
    v14 = v28;
LABEL_32:
    sub_9183A0((__int64)&v27, v14, v30);
    v16 = v28;
    goto LABEL_19;
  }
  if ( v27 )
  {
    *v27 = v11;
    v14 = v29;
    v12 = v28;
  }
  v15 = v12 + 1;
  v30[0] = v10;
  v28 = v15;
  if ( v15 == v14 )
    goto LABEL_32;
LABEL_17:
  *v15 = v10;
  v15 = v28;
LABEL_18:
  v16 = v15 + 1;
  v28 = v16;
LABEL_19:
  a1[1] = sub_BD0EC0(v27, v16 - v27, "gc_stackentry", 0xDu, 0);
  v17 = sub_BA8CD0((__int64)a2, (__int64)"llvm_gc_root_chain", 0x12u, 0);
  *a1 = (__int64)v17;
  if ( v17 )
  {
    if ( (v17[32] & 0xF) == 0 && sub_B2FC80((__int64)v17) )
    {
      v19 = *a1;
      v20 = sub_AD6530((__int64)v13, (__int64)"llvm_gc_root_chain");
      sub_B30160(v19, v20);
      v21 = *a1;
      v22 = *(_BYTE *)(*a1 + 32) & 0xF0 | 2;
      *(_BYTE *)(*a1 + 32) = v22;
      if ( (v22 & 0x30) != 0 )
        *(_BYTE *)(v21 + 33) |= 0x40u;
    }
  }
  else
  {
    v23 = sub_AD6530((__int64)v13, (__int64)"llvm_gc_root_chain");
    v30[0] = "llvm_gc_root_chain";
    v24 = v23;
    v32 = 1;
    v31 = 3;
    BYTE4(v26) = 0;
    v25 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v25 )
      sub_B30000((__int64)v25, (__int64)a2, v13, 0, 2, v24, (__int64)v30, 0, 0, v26, 0);
    *a1 = (__int64)v25;
  }
  if ( v27 )
    j_j___libc_free_0((unsigned __int64)v27);
  return 1;
}
