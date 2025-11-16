// Function: sub_9CE5C0
// Address: 0x9ce5c0
//
__int64 *__fastcall sub_9CE5C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rcx
  int v6; // ebx
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // r15
  __int64 v14; // r13
  bool v15; // dl
  unsigned __int64 *v16; // rdi
  __int64 v17; // r8
  bool v18; // zf
  int v19; // r15d
  __int64 v20; // r13
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v25; // [rsp+28h] [rbp-A8h]
  _QWORD v26[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v27[4]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v28; // [rsp+60h] [rbp-70h] BYREF
  __int64 v29; // [rsp+68h] [rbp-68h]
  _QWORD v30[12]; // [rsp+70h] [rbp-60h] BYREF

  sub_9C66D0((__int64)&v28, a2, 4, a4);
  v5 = v29 & 1;
  LOBYTE(v29) = v29 & 0xFD;
  if ( (_BYTE)v5 )
    goto LABEL_25;
  v6 = 3;
  LOBYTE(v25) = v25 & 0xFC;
  LODWORD(v24) = v28;
  if ( (v28 & 8) == 0 )
  {
LABEL_3:
    v7 = *(unsigned int *)(a2 + 32);
    if ( (unsigned int)v7 > 0x1F )
    {
      v7 = (unsigned int)(v7 - 32);
      *(_DWORD *)(a2 + 32) = 32;
      *(_QWORD *)(a2 + 24) >>= v7;
    }
    else
    {
      *(_DWORD *)(a2 + 32) = 0;
    }
    sub_9C66D0((__int64)&v28, a2, 32, v7);
    if ( (v29 & 1) != 0 )
      goto LABEL_25;
    v9 = *(_QWORD *)(a2 + 16);
    v10 = *(unsigned int *)(a2 + 32);
    v11 = 8 * v9 - v10;
    v12 = *(_QWORD *)(a2 + 8);
    v13 = v11 + 32LL * (unsigned int)v28;
    if ( (_DWORD)v10 || v9 < v12 )
    {
      if ( v13 >> 3 <= v12 )
      {
        sub_9CDFE0((__int64 *)&v28, a2, v11 + 32LL * (unsigned int)v28, v12);
        v23 = v28 | 1;
        if ( (v28 & 0xFFFFFFFFFFFFFFFELL) == 0 )
          v23 = 1;
        *a1 = v23;
      }
      else
      {
        v22 = sub_2241E50(&v28, a2, v9, v12, v8);
        v24 = (unsigned __int64)v26;
        v30[3] = 0x100000000LL;
        v30[4] = &v24;
        v28 = (unsigned __int64)&unk_49DD210;
        v25 = 0;
        LOBYTE(v26[0]) = 0;
        v29 = 0;
        memset(v30, 0, 24);
        sub_CB5980(&v28, 0, 0, 0);
        v27[2] = v11;
        v27[1] = "can't skip to bit %zu from %lu";
        v27[3] = v13;
        v27[0] = &unk_49D98C0;
        sub_CB6620(&v28, v27);
        v28 = (unsigned __int64)&unk_49DD210;
        sub_CB5840(&v28);
        sub_9C3320(a1, (__int64)&v24, 0x54u, v22);
        if ( (_QWORD *)v24 != v26 )
          j_j___libc_free_0(v24, v26[0] + 1LL);
      }
    }
    else
    {
      v14 = sub_2241E50(&v28, a2, v9, v12, v8);
      v28 = (unsigned __int64)v30;
      sub_9C2D70((__int64 *)&v28, "can't skip block: already at end of stream", (__int64)"");
      sub_C63F00(a1, &v28, 84, v14);
      if ( (_QWORD *)v28 != v30 )
        j_j___libc_free_0(v28, v30[0] + 1LL);
    }
    return a1;
  }
  do
  {
    v16 = &v28;
    sub_9C66D0((__int64)&v28, a2, 4, v5);
    v18 = (v29 & 1) == 0;
    v19 = v29 & 1;
    LOBYTE(v29) = v29 & 0xFD;
    if ( v18 )
    {
      LODWORD(v27[0]) = v28;
      v15 = v25 & 1;
      if ( (v25 & 1) == 0 )
        goto LABEL_11;
    }
    else
    {
      v27[0] = v28;
      v28 = 0;
      v15 = v25 & 1;
      if ( (v25 & 1) == 0 )
        goto LABEL_11;
    }
    v16 = (unsigned __int64 *)v24;
    v15 = 0;
    if ( v24 )
    {
      (*(void (__fastcall **)(unsigned __int64, __int64, _QWORD))(*(_QWORD *)v24 + 8LL))(v24, a2, 0);
      v15 = (v29 & 2) != 0;
    }
LABEL_11:
    LOBYTE(v25) = v19 | v25 & 0xFE | 2;
    if ( (_BYTE)v19 )
      v24 = v27[0];
    else
      LODWORD(v24) = v27[0];
    if ( v15 )
      sub_9CDF70(&v28);
    if ( (v29 & 1) != 0 )
    {
      v16 = (unsigned __int64 *)v28;
      if ( v28 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v28 + 8LL))(v28);
    }
    LOBYTE(v25) = (2 * v19) | v25 & 0xFD;
    if ( (_BYTE)v19 )
    {
      v28 = v24;
      goto LABEL_25;
    }
    if ( (v24 & 8) == 0 )
      goto LABEL_3;
    v6 += 3;
  }
  while ( v6 != 33 );
  v20 = sub_2241E50(v16, a2, (unsigned int)(2 * v19), v5, v17);
  v28 = (unsigned __int64)v30;
  sub_9C2D70((__int64 *)&v28, "Unterminated VBR", (__int64)"");
  sub_C63F00(v27, &v28, 84, v20);
  if ( (_QWORD *)v28 != v30 )
    j_j___libc_free_0(v28, v30[0] + 1LL);
  v28 = v27[0] & 0xFFFFFFFFFFFFFFFELL;
LABEL_25:
  *a1 = v28 | 1;
  return a1;
}
