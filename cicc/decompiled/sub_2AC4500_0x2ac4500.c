// Function: sub_2AC4500
// Address: 0x2ac4500
//
__int64 __fastcall sub_2AC4500(__int64 a1, __int64 a2, char a3, __int64 *a4)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // rdx
  void *v12; // rax
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rbx
  __int64 v20; // r8
  __int64 v21; // rcx
  __int64 v22; // rdx
  void **v23; // r13
  __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  void *v26; // rax
  char v27; // dl
  unsigned __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r9
  __int64 v34; // r12
  unsigned __int64 v35; // rcx
  char v37; // [rsp+1Fh] [rbp-F1h]
  __int64 v38; // [rsp+30h] [rbp-E0h]
  __int64 v40; // [rsp+40h] [rbp-D0h]
  void *v42; // [rsp+78h] [rbp-98h] BYREF
  void *v43; // [rsp+80h] [rbp-90h] BYREF
  void *v44; // [rsp+88h] [rbp-88h] BYREF
  void *v45; // [rsp+90h] [rbp-80h] BYREF
  void *v46; // [rsp+98h] [rbp-78h] BYREF
  void *v47; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-68h]
  void *v49[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v50; // [rsp+D0h] [rbp-40h]

  v5 = sub_AD64C0(a2, 0, 0);
  v6 = sub_2AC42A0(a1, v5);
  v45 = (void *)*a4;
  if ( v45 )
    sub_2AAAFA0((__int64 *)&v45);
  v7 = sub_22077B0(0x98u);
  if ( !v7 )
  {
    v13 = 96;
    v10 = 40;
    goto LABEL_13;
  }
  v46 = v45;
  if ( v45 )
  {
    sub_2AAAFA0((__int64 *)&v46);
    v47 = v46;
    if ( v46 )
    {
      sub_2AAAFA0((__int64 *)&v47);
      v49[0] = v47;
      if ( v47 )
        sub_2AAAFA0((__int64 *)v49);
      goto LABEL_8;
    }
  }
  else
  {
    v47 = 0;
  }
  v49[0] = 0;
LABEL_8:
  *(_BYTE *)(v7 + 8) = 29;
  v10 = v7 + 40;
  *(_QWORD *)(v7 + 24) = 0;
  *(_QWORD *)(v7 + 32) = 0;
  *(_QWORD *)v7 = &unk_4A231A8;
  *(_QWORD *)(v7 + 16) = 0;
  *(_QWORD *)(v7 + 64) = v6;
  *(_QWORD *)(v7 + 40) = &unk_4A23170;
  *(_QWORD *)(v7 + 48) = v7 + 64;
  *(_QWORD *)(v7 + 56) = 0x200000001LL;
  v11 = *(unsigned int *)(v6 + 24);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 28) )
  {
    sub_C8D5F0(v6 + 16, (const void *)(v6 + 32), v11 + 1, 8u, v8, v9);
    v11 = *(unsigned int *)(v6 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(v6 + 16) + 8 * v11) = v10;
  ++*(_DWORD *)(v6 + 24);
  *(_QWORD *)(v7 + 80) = 0;
  *(_QWORD *)(v7 + 40) = &unk_4A23AA8;
  v12 = v49[0];
  *(_QWORD *)v7 = &unk_4A23A70;
  *(_QWORD *)(v7 + 88) = v12;
  if ( v12 )
    sub_2AAAFA0((__int64 *)(v7 + 88));
  v13 = v7 + 96;
  sub_9C6650(v49);
  sub_2BF0340(v7 + 96, 1, 0, v7);
  *(_QWORD *)v7 = &unk_4A231C8;
  *(_QWORD *)(v7 + 40) = &unk_4A23200;
  *(_QWORD *)(v7 + 96) = &unk_4A23238;
  sub_9C6650(&v47);
  *(_QWORD *)v7 = &unk_4A23FE8;
  *(_QWORD *)(v7 + 40) = &unk_4A24030;
  *(_QWORD *)(v7 + 96) = &unk_4A24068;
  sub_9C6650(&v46);
  *(_QWORD *)v7 = &unk_4A23390;
  *(_QWORD *)(v7 + 40) = &unk_4A233E8;
  *(_QWORD *)(v7 + 96) = &unk_4A23420;
LABEL_13:
  sub_9C6650(&v45);
  v14 = sub_2BF3F10(a1);
  v15 = sub_2BF04D0(v14);
  v16 = *(__int64 **)(v15 + 120);
  *(_QWORD *)(v7 + 80) = v15;
  v17 = *(_QWORD *)(v7 + 24);
  v18 = *v16;
  *(_QWORD *)(v7 + 32) = v16;
  v18 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v7 + 24) = v18 | v17 & 7;
  *(_QWORD *)(v18 + 8) = v7 + 24;
  *v16 = *v16 & 7 | (v7 + 24);
  v40 = sub_2BF0520(v14);
  v38 = v40 + 112;
  v49[0] = "index.next";
  v50 = 259;
  v42 = (void *)*a4;
  if ( v42 )
  {
    sub_2AAAFA0((__int64 *)&v42);
    v47 = (void *)v13;
    v43 = v42;
    v48 = a1 + 328;
    v37 = (char)(a3 << 7) >> 7;
    if ( v42 )
      sub_2AAAFA0((__int64 *)&v43);
  }
  else
  {
    v47 = (void *)v13;
    v43 = 0;
    v37 = (char)(a3 << 7) >> 7;
    v48 = a1 + 328;
  }
  v19 = sub_22077B0(0xC8u);
  if ( v19 )
  {
    v44 = v43;
    if ( v43 )
    {
      sub_2AAAFA0((__int64 *)&v44);
      v45 = v44;
      if ( v44 )
      {
        sub_2AAAFA0((__int64 *)&v45);
        v46 = v45;
        if ( v45 )
          sub_2AAAFA0((__int64 *)&v46);
        goto LABEL_21;
      }
    }
    else
    {
      v45 = 0;
    }
    v46 = 0;
LABEL_21:
    v20 = v19 + 64;
    *(_BYTE *)(v19 + 8) = 4;
    v21 = v19 + 64;
    v22 = 0;
    *(_QWORD *)(v19 + 24) = 0;
    *(_QWORD *)v19 = &unk_4A231A8;
    *(_QWORD *)(v19 + 32) = 0;
    *(_QWORD *)(v19 + 16) = 0;
    *(_QWORD *)(v19 + 40) = &unk_4A23170;
    *(_QWORD *)(v19 + 56) = 0x200000000LL;
    v23 = &v47;
    for ( *(_QWORD *)(v19 + 48) = v19 + 64; ; v21 = *(_QWORD *)(v19 + 48) )
    {
      *(_QWORD *)(v21 + 8 * v22) = v13;
      v24 = *(unsigned int *)(v13 + 24);
      v25 = *(unsigned int *)(v13 + 28);
      ++*(_DWORD *)(v19 + 56);
      if ( v24 + 1 > v25 )
      {
        sub_C8D5F0(v13 + 16, (const void *)(v13 + 32), v24 + 1, 8u, v20, v24 + 1);
        v24 = *(unsigned int *)(v13 + 24);
      }
      ++v23;
      *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8 * v24) = v19 + 40;
      ++*(_DWORD *)(v13 + 24);
      if ( v23 == v49 )
        break;
      v22 = *(unsigned int *)(v19 + 56);
      v13 = (__int64)*v23;
      if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v19 + 60) )
      {
        sub_C8D5F0(v19 + 48, (const void *)(v19 + 64), v22 + 1, 8u, v20, v22 + 1);
        v22 = *(unsigned int *)(v19 + 56);
      }
    }
    *(_QWORD *)(v19 + 80) = 0;
    *(_QWORD *)(v19 + 40) = &unk_4A23AA8;
    v26 = v46;
    *(_QWORD *)v19 = &unk_4A23A70;
    *(_QWORD *)(v19 + 88) = v26;
    if ( v26 )
      sub_2AAAFA0((__int64 *)(v19 + 88));
    sub_9C6650(&v46);
    sub_2BF0340(v19 + 96, 1, 0, v19);
    *(_QWORD *)v19 = &unk_4A231C8;
    *(_QWORD *)(v19 + 40) = &unk_4A23200;
    *(_QWORD *)(v19 + 96) = &unk_4A23238;
    sub_9C6650(&v45);
    *(_BYTE *)(v19 + 152) = 1;
    *(_QWORD *)v19 = &unk_4A23258;
    *(_QWORD *)(v19 + 40) = &unk_4A23290;
    v27 = *(_BYTE *)(v19 + 156);
    *(_QWORD *)(v19 + 96) = &unk_4A232C8;
    *(_BYTE *)(v19 + 156) = v37 & 1 | v27 & 0xFC;
    sub_9C6650(&v44);
    *(_BYTE *)(v19 + 160) = 13;
    *(_QWORD *)v19 = &unk_4A23B70;
    *(_QWORD *)(v19 + 40) = &unk_4A23BB8;
    *(_QWORD *)(v19 + 96) = &unk_4A23BF0;
    sub_CA0F50((__int64 *)(v19 + 168), v49);
  }
  if ( v40 )
  {
    *(_QWORD *)(v19 + 80) = v40;
    v28 = *(_QWORD *)(v40 + 112) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v19 + 32) = v38;
    *(_QWORD *)(v19 + 24) = v28 | *(_QWORD *)(v19 + 24) & 7LL;
    *(_QWORD *)(v28 + 8) = v19 + 24;
    *(_QWORD *)(v40 + 112) = *(_QWORD *)(v40 + 112) & 7LL | (v19 + 24);
    sub_9C6650(&v43);
    sub_9C6650(&v42);
LABEL_33:
    v19 += 96;
    goto LABEL_34;
  }
  sub_9C6650(&v43);
  sub_9C6650(&v42);
  if ( v19 )
    goto LABEL_33;
LABEL_34:
  sub_2AAECA0(v10, v19, v29, v30, v31, v32);
  v50 = 257;
  v42 = (void *)*a4;
  if ( !v42 )
  {
    v47 = (void *)v19;
    v43 = 0;
    v48 = a1 + 216;
    goto LABEL_50;
  }
  sub_2AAAFA0((__int64 *)&v42);
  v47 = (void *)v19;
  v43 = v42;
  v48 = a1 + 216;
  if ( !v42 )
  {
LABEL_50:
    v44 = 0;
    goto LABEL_38;
  }
  sub_2AAAFA0((__int64 *)&v43);
  v44 = v43;
  if ( v43 )
    sub_2AAAFA0((__int64 *)&v44);
LABEL_38:
  v34 = sub_22077B0(0xC8u);
  if ( v34 )
  {
    v45 = v44;
    if ( v44 )
    {
      sub_2AAAFA0((__int64 *)&v45);
      v46 = v45;
      if ( v45 )
        sub_2AAAFA0((__int64 *)&v46);
    }
    else
    {
      v46 = 0;
    }
    sub_2AAF4A0(v34, 4, (__int64 *)&v47, 2, (__int64 *)&v46, v33);
    sub_9C6650(&v46);
    *(_BYTE *)(v34 + 152) = 7;
    *(_DWORD *)(v34 + 156) = 0;
    *(_QWORD *)v34 = &unk_4A23258;
    *(_QWORD *)(v34 + 40) = &unk_4A23290;
    *(_QWORD *)(v34 + 96) = &unk_4A232C8;
    sub_9C6650(&v45);
    *(_BYTE *)(v34 + 160) = 78;
    *(_QWORD *)v34 = &unk_4A23B70;
    *(_QWORD *)(v34 + 40) = &unk_4A23BB8;
    *(_QWORD *)(v34 + 96) = &unk_4A23BF0;
    sub_CA0F50((__int64 *)(v34 + 168), v49);
  }
  if ( v40 )
  {
    *(_QWORD *)(v34 + 80) = v40;
    v35 = *(_QWORD *)(v40 + 112) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v34 + 32) = v38;
    *(_QWORD *)(v34 + 24) = v35 | *(_QWORD *)(v34 + 24) & 7LL;
    *(_QWORD *)(v35 + 8) = v34 + 24;
    *(_QWORD *)(v40 + 112) = *(_QWORD *)(v40 + 112) & 7LL | (v34 + 24);
  }
  sub_9C6650(&v44);
  sub_9C6650(&v43);
  return sub_9C6650(&v42);
}
