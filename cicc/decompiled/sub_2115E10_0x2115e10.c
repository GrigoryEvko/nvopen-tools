// Function: sub_2115E10
// Address: 0x2115e10
//
__int64 __fastcall sub_2115E10(_QWORD *a1, __int64 *a2)
{
  __int64 *v3; // r12
  __int64 *v4; // rbx
  __int64 v5; // rax
  _QWORD *v6; // r8
  void *v7; // rdx
  int v9; // eax
  _QWORD *v10; // rdi
  void *v11; // rax
  __int64 **v12; // rsi
  __int64 **v13; // rsi
  __int64 *v14; // rax
  void *v15; // rbx
  __int64 *v16; // rax
  void *v17; // rax
  __int64 **v18; // rsi
  __int64 **v19; // rdx
  __int64 **v20; // rsi
  __int64 **v21; // rsi
  __int64 **v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // r14
  _QWORD *v34; // rax
  _QWORD *v35; // rbx
  _QWORD *v36; // [rsp+0h] [rbp-80h]
  __int64 **v38; // [rsp+10h] [rbp-70h] BYREF
  __int64 **v39; // [rsp+18h] [rbp-68h]
  __int64 **v40; // [rsp+20h] [rbp-60h]
  void *s2[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v42[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a2 + 3;
  v4 = (__int64 *)a2[4];
  if ( v4 == a2 + 3 )
    return 0;
  while ( 1 )
  {
    if ( !v4 )
      BUG();
    if ( (*((_BYTE *)v4 - 37) & 0x40) == 0 )
      goto LABEL_3;
    s2[0] = v42;
    sub_2113CA0((__int64 *)s2, "shadow-stack", (__int64)"");
    v5 = sub_15E0FA0((__int64)(v4 - 7));
    v6 = s2[0];
    v7 = *(void **)(v5 + 8);
    if ( v7 == s2[1] )
    {
      if ( !v7 )
        break;
      v36 = s2[0];
      v9 = memcmp(*(const void **)v5, s2[0], (size_t)v7);
      v6 = v36;
      if ( !v9 )
        break;
    }
    if ( v6 == v42 )
    {
LABEL_3:
      v4 = (__int64 *)v4[1];
      if ( v3 == v4 )
        return 0;
    }
    else
    {
      j_j___libc_free_0(v6, v42[0] + 1LL);
      v4 = (__int64 *)v4[1];
      if ( v3 == v4 )
        return 0;
    }
  }
  if ( v6 != v42 )
    j_j___libc_free_0(v6, v42[0] + 1LL);
  v10 = (_QWORD *)*a2;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  s2[0] = (void *)sub_1643350(v10);
  sub_1278040((__int64)&v38, 0, s2);
  v11 = (void *)sub_1643350((_QWORD *)*a2);
  v12 = v39;
  s2[0] = v11;
  if ( v39 == v40 )
  {
    sub_1278040((__int64)&v38, v39, s2);
    v13 = v39;
  }
  else
  {
    if ( v39 )
    {
      *v39 = (__int64 *)v11;
      v12 = v39;
    }
    v13 = v12 + 1;
    v39 = v13;
  }
  v14 = (__int64 *)sub_1644140(v38, v13 - v38, "gc_map", 6u, 0);
  a1[22] = v14;
  v15 = (void *)sub_1646BA0(v14, 0);
  v16 = (__int64 *)sub_1644060(*a2, "gc_stackentry", 0xDu);
  a1[21] = v16;
  if ( v38 != v39 )
    v39 = v38;
  v17 = (void *)sub_1646BA0(v16, 0);
  v18 = v39;
  v19 = v40;
  s2[0] = v17;
  if ( v39 == v40 )
  {
    sub_1278040((__int64)&v38, v39, s2);
    s2[0] = v15;
    v20 = v39;
    if ( v39 == v40 )
      goto LABEL_37;
    if ( v39 )
      goto LABEL_24;
LABEL_25:
    v21 = v20 + 1;
    v39 = v21;
  }
  else
  {
    if ( v39 )
    {
      *v39 = (__int64 *)v17;
      v18 = v39;
      v19 = v40;
    }
    v20 = v18 + 1;
    s2[0] = v15;
    v39 = v20;
    if ( v20 != v19 )
    {
LABEL_24:
      *v20 = (__int64 *)v15;
      v20 = v39;
      goto LABEL_25;
    }
LABEL_37:
    sub_1278040((__int64)&v38, v20, s2);
    v21 = v39;
  }
  sub_1643FB0(a1[21], v38, v21 - v38, 0);
  v22 = (__int64 **)sub_1646BA0((__int64 *)a1[21], 0);
  v23 = sub_16321C0((__int64)a2, (__int64)"llvm_gc_root_chain", 18, 0);
  a1[20] = v23;
  if ( v23 )
  {
    if ( (*(_BYTE *)(v23 + 32) & 0xF) == 0 && sub_15E4F60(v23) )
    {
      v28 = a1[20];
      v29 = sub_15A06D0(v22, (__int64)"llvm_gc_root_chain", v26, v27);
      sub_15E5440(v28, v29);
      v30 = a1[20];
      v31 = *(_BYTE *)(v30 + 32) & 0xF0 | 2;
      *(_BYTE *)(v30 + 32) = v31;
      if ( (v31 & 0x30) != 0 )
        *(_BYTE *)(v30 + 33) |= 0x40u;
    }
  }
  else
  {
    v32 = sub_15A06D0(v22, (__int64)"llvm_gc_root_chain", v24, v25);
    s2[0] = "llvm_gc_root_chain";
    v33 = v32;
    LOWORD(v42[0]) = 259;
    v34 = sub_1648A60(88, 1u);
    v35 = v34;
    if ( v34 )
      sub_15E51E0((__int64)v34, (__int64)a2, (__int64)v22, 0, 2, v33, (__int64)s2, 0, 0, 0, 0);
    a1[20] = v35;
  }
  if ( v38 )
    j_j___libc_free_0(v38, (char *)v40 - (char *)v38);
  return 1;
}
