// Function: sub_2AC5550
// Address: 0x2ac5550
//
__int64 __fastcall sub_2AC5550(__int64 *a1, unsigned __int8 *a2, __int64 *a3)
{
  __int64 v5; // rbx
  char v6; // al
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // r12
  __int64 v13; // r8
  __int64 v14; // rdx
  char **v15; // r14
  const void *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  _QWORD *v41; // r14
  __int64 v42; // rax
  __int64 v43; // r9
  int v44; // eax
  __int64 v45; // r14
  _QWORD *v46; // rax
  const void *v47; // [rsp+0h] [rbp-90h]
  __int64 v48; // [rsp+0h] [rbp-90h]
  __int64 v49; // [rsp+8h] [rbp-88h]
  const void *v50; // [rsp+8h] [rbp-88h]
  __int64 v51; // [rsp+8h] [rbp-88h]
  int v52; // [rsp+10h] [rbp-80h]
  char *v53; // [rsp+10h] [rbp-80h]
  __int64 v54; // [rsp+18h] [rbp-78h]
  __int64 v55; // [rsp+18h] [rbp-78h]
  __int64 v56; // [rsp+20h] [rbp-70h] BYREF
  __int64 v57; // [rsp+28h] [rbp-68h] BYREF
  char *v58; // [rsp+30h] [rbp-60h] BYREF
  __int64 v59; // [rsp+38h] [rbp-58h]
  _BYTE v60[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v61; // [rsp+50h] [rbp-40h]

  v5 = *a3;
  v54 = a3[1];
  v6 = *(_BYTE *)(sub_2BF0490(*a3) + 8);
  if ( v6 == 8 || v6 == 36 )
  {
    v29 = v5;
    v5 = v54;
    v54 = v29;
    LODWORD(v29) = *a2;
    v52 = v29 - 29;
    if ( (_DWORD)v29 != 44 )
      goto LABEL_4;
  }
  else
  {
    v7 = *a2;
    v52 = v7 - 29;
    if ( v7 != 44 )
      goto LABEL_4;
  }
  v30 = sub_AD64C0(*((_QWORD *)a2 + 1), 0, 0);
  v31 = *a1;
  v58 = v60;
  v59 = 0x200000000LL;
  v32 = sub_2AC42A0(v31, v30);
  sub_2AB9420((__int64)&v58, v32, v33, v34, v35, v36);
  sub_2AB9420((__int64)&v58, v5, v37, v38, v39, v40);
  v41 = v58;
  v53 = &v58[8 * (unsigned int)v59];
  v42 = sub_22077B0(0xA8u);
  v5 = v42;
  if ( v42 )
  {
    v5 = v42 + 96;
    sub_2ABDBC0(v42, 23, v41, v53, a2, v43);
    *(_QWORD *)(v5 - 96) = &unk_4A23EC8;
    *(_QWORD *)v5 = &unk_4A23F38;
    v44 = *a2;
    *(_QWORD *)(v5 - 56) = &unk_4A23F00;
    *(_DWORD *)(v5 + 64) = v44 - 29;
  }
  v45 = a1[7];
  v46 = (_QWORD *)sub_2BF0490(v5);
  sub_2AAFF40(*(_QWORD *)v45, v46, *(unsigned __int64 **)(v45 + 8));
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  v52 = 13;
LABEL_4:
  v8 = a1[5];
  v9 = *((_QWORD *)a2 + 5);
  if ( !*(_BYTE *)(v8 + 108) || !*(_DWORD *)(v8 + 100) )
  {
    if ( !(unsigned __int8)sub_31A6C30(*(_QWORD *)(v8 + 440), v9) )
      goto LABEL_7;
    v9 = *((_QWORD *)a2 + 5);
  }
  v21 = sub_2AB6F10((__int64)a1, v9);
  v22 = *a1;
  v23 = v21;
  v24 = sub_AD64C0(*((_QWORD *)a2 + 1), 0, 0);
  v25 = sub_2AC42A0(v22, v24);
  v26 = (__int64 *)a1[7];
  v27 = v25;
  v28 = *((_QWORD *)a2 + 6);
  v61 = 257;
  v56 = v28;
  if ( v28 )
  {
    v51 = v27;
    sub_2AAAFA0(&v56);
    v27 = v51;
  }
  v5 = sub_2AAF680(v26, v23, v5, v27, &v56, (void **)&v58, v57, 0);
  sub_9C6650(&v56);
LABEL_7:
  v10 = sub_22077B0(0xA0u);
  v12 = v10;
  if ( v10 )
  {
    v58 = (char *)v5;
    v13 = v10 + 40;
    v14 = 0;
    v56 = 0;
    v15 = &v58;
    v59 = v54;
    *(_QWORD *)(v10 + 56) = 0x200000000LL;
    *(_BYTE *)(v10 + 8) = 8;
    *(_QWORD *)v10 = &unk_4A231A8;
    v57 = 0;
    v55 = v10 + 48;
    *(_QWORD *)(v10 + 40) = &unk_4A23170;
    v16 = (const void *)(v10 + 64);
    *(_QWORD *)(v12 + 24) = 0;
    v17 = v12 + 64;
    *(_QWORD *)(v12 + 32) = 0;
    *(_QWORD *)(v12 + 16) = 0;
    for ( *(_QWORD *)(v12 + 48) = v12 + 64; ; v17 = *(_QWORD *)(v12 + 48) )
    {
      *(_QWORD *)(v17 + 8 * v14) = v5;
      ++*(_DWORD *)(v12 + 56);
      v18 = *(unsigned int *)(v5 + 24);
      if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 28) )
      {
        v47 = v16;
        v49 = v13;
        sub_C8D5F0(v5 + 16, (const void *)(v5 + 32), v18 + 1, 8u, v13, v11);
        v18 = *(unsigned int *)(v5 + 24);
        v16 = v47;
        v13 = v49;
      }
      ++v15;
      *(_QWORD *)(*(_QWORD *)(v5 + 16) + 8 * v18) = v13;
      ++*(_DWORD *)(v5 + 24);
      if ( v15 == (char **)v60 )
        break;
      v14 = *(unsigned int *)(v12 + 56);
      v5 = (__int64)*v15;
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 60) )
      {
        v48 = v13;
        v50 = v16;
        sub_C8D5F0(v55, v16, v14 + 1, 8u, v13, v11);
        v14 = *(unsigned int *)(v12 + 56);
        v13 = v48;
        v16 = v50;
      }
    }
    *(_QWORD *)(v12 + 80) = 0;
    *(_QWORD *)(v12 + 40) = &unk_4A23AA8;
    v19 = v57;
    *(_QWORD *)v12 = &unk_4A23A70;
    *(_QWORD *)(v12 + 88) = v19;
    if ( v19 )
    {
      sub_2AAAFA0((__int64 *)(v12 + 88));
      if ( v57 )
        sub_B91220((__int64)&v57, v57);
    }
    sub_2BF0340(v12 + 96, 1, a2, v12);
    *(_QWORD *)v12 = &unk_4A231C8;
    *(_QWORD *)(v12 + 40) = &unk_4A23200;
    *(_QWORD *)(v12 + 96) = &unk_4A23238;
    sub_9C6650(&v56);
    *(_QWORD *)v12 = &unk_4A23AE0;
    *(_QWORD *)(v12 + 96) = &unk_4A23B50;
    *(_QWORD *)(v12 + 40) = &unk_4A23B18;
    *(_DWORD *)(v12 + 152) = v52;
    sub_2BF0490(*(_QWORD *)(*(_QWORD *)(v12 + 48) + 8LL));
  }
  return v12;
}
