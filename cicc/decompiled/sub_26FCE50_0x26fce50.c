// Function: sub_26FCE50
// Address: 0x26fce50
//
__int64 __fastcall sub_26FCE50(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  char *v5; // r12
  __int16 v6; // cx
  char v7; // cl
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  char *v13; // rdx
  char *v14; // r12
  char v15; // cl
  __int64 *v16; // rax
  __int64 **v17; // rax
  __int64 v18; // rax
  char *v19; // r15
  __int64 v20; // r12
  size_t v21; // r12
  __int64 *v22; // rax
  __int64 **v23; // rax
  __int64 **v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // r15
  char v28; // cl
  unsigned __int8 *v29; // rax
  unsigned __int8 *v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rcx
  char v35; // dl
  __int16 v36; // ax
  unsigned __int16 v37; // ax
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // r10
  __int64 v43; // r8
  unsigned __int8 *v44; // r12
  unsigned __int8 v45; // al
  char v46; // al
  char v47; // [rsp+Fh] [rbp-81h]
  _QWORD *v48; // [rsp+10h] [rbp-80h]
  __int64 v49; // [rsp+10h] [rbp-80h]
  __int64 v50; // [rsp+10h] [rbp-80h]
  char v51; // [rsp+18h] [rbp-78h]
  __int64 v52; // [rsp+18h] [rbp-78h]
  __int64 v53; // [rsp+18h] [rbp-78h]
  __int64 v54[2]; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v55; // [rsp+30h] [rbp-60h] BYREF
  __int64 v56; // [rsp+38h] [rbp-58h]
  unsigned __int64 v57; // [rsp+40h] [rbp-50h]
  unsigned int v58; // [rsp+48h] [rbp-48h]
  __int16 v59; // [rsp+50h] [rbp-40h]

  result = a2[3];
  v5 = (char *)a2[2];
  if ( v5 != (char *)result || a2[8] != a2[9] )
  {
    v6 = (*(_WORD *)(*a2 + 34) >> 1) & 0x3F;
    if ( v6 )
    {
      v7 = v6 - 1;
    }
    else
    {
      v46 = sub_AE5020(*a1 + 312, *(_QWORD *)(*a2 + 24));
      v5 = (char *)a2[2];
      v7 = v46;
      result = a2[3];
    }
    v8 = result - (_QWORD)v5;
    v9 = (result - (_QWORD)v5 + (1LL << v7) - 1) & -(1LL << v7);
    if ( result - (__int64)v5 < v9 )
    {
      sub_CD93F0(a2 + 2, v9 - v8);
      v5 = (char *)a2[2];
      v8 = a2[3] - (_QWORD)v5;
    }
    else if ( result - (__int64)v5 > v9 && (char *)result != &v5[v9] )
    {
      a2[3] = (__int64)&v5[v9];
      v8 = (result - (_QWORD)v5 + (1LL << v7) - 1) & -(1LL << v7);
    }
    v10 = v8 >> 1;
    if ( v8 >> 1 )
    {
      v11 = v8 - 1;
      v12 = 0;
      do
      {
        v13 = &v5[v11 - v12];
        v14 = &v5[v12++];
        v15 = *v14;
        *v14 = *v13;
        *v13 = v15;
        v5 = (char *)a2[2];
      }
      while ( v12 != v10 );
      v8 = a2[3] - (_QWORD)v5;
    }
    v16 = (__int64 *)sub_BCD140(*(_QWORD **)*a1, 8u);
    v17 = (__int64 **)sub_BCD420(v16, v8);
    v18 = sub_AC9630(v5, v8, v17);
    v19 = (char *)a2[8];
    v20 = a2[9];
    v55 = v18;
    v21 = v20 - (_QWORD)v19;
    v56 = *(_QWORD *)(*a2 - 32);
    v22 = (__int64 *)sub_BCD140(*(_QWORD **)*a1, 8u);
    v23 = (__int64 **)sub_BCD420(v22, v21);
    v57 = sub_AC9630(v19, v21, v23);
    v24 = (__int64 **)sub_AC34C0(&v55, 3, 0);
    v25 = sub_AD24A0(v24, (__int64 *)&v55, 3);
    v26 = *(_QWORD **)(v25 + 8);
    v27 = v25;
    v28 = *(_BYTE *)(*a2 + 80);
    v59 = 257;
    v48 = v26;
    BYTE4(v54[0]) = 0;
    v51 = v28 & 1;
    v29 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
    v30 = v29;
    if ( v29 )
      sub_B30000((__int64)v29, *a1, v48, v51, 8, v27, (__int64)&v55, *a2, 0, v54[0], 0);
    v31 = 0;
    v32 = 0;
    if ( (*(_BYTE *)(*a2 + 35) & 4) != 0 )
      v32 = sub_B31D10(*a2, 0, 0);
    sub_B31A00((__int64)v30, v32, v31);
    sub_B2F990((__int64)v30, *(_QWORD *)(*a2 + 48), v33, v34);
    v35 = 0;
    v36 = (*(_WORD *)(*a2 + 34) >> 1) & 0x3F;
    if ( v36 )
    {
      v35 = 1;
      v47 = v36 - 1;
    }
    LOBYTE(v37) = v47;
    HIBYTE(v37) = v35;
    sub_B2F740((__int64)v30, v37);
    sub_B9E560((__int64)v30, *a2, *((_DWORD *)a2 + 6) - *((_DWORD *)a2 + 4));
    v52 = *a1;
    v38 = sub_ACD640(a1[9], 0, 0);
    v39 = a1[9];
    v54[0] = v38;
    v54[1] = sub_ACD640(v39, 1, 0);
    v40 = *(_QWORD *)(v27 + 8);
    LOBYTE(v59) = 0;
    v41 = sub_AD9FD0(v40, v30, v54, 2, 3u, (__int64)&v55, 0);
    v42 = v52;
    v43 = v41;
    if ( (_BYTE)v59 )
    {
      LOBYTE(v59) = 0;
      if ( v58 > 0x40 && v57 )
      {
        v49 = v41;
        j_j___libc_free_0_0(v57);
        v43 = v49;
        v42 = v52;
      }
      if ( (unsigned int)v56 > 0x40 && v55 )
      {
        v50 = v43;
        v53 = v42;
        j_j___libc_free_0_0(v55);
        v43 = v50;
        v42 = v53;
      }
    }
    v59 = 257;
    v44 = (unsigned __int8 *)sub_B30500(
                               *(_QWORD **)(*(_QWORD *)(*a2 - 32) + 8LL),
                               0,
                               *(_BYTE *)(*a2 + 32) & 0xF,
                               (__int64)&v55,
                               v43,
                               v42);
    v45 = *(_BYTE *)(*a2 + 32) & 0x30 | v44[32] & 0xCF;
    v44[32] = v45;
    if ( (v45 & 0xFu) - 7 <= 1 || (v45 & 0x30) != 0 && (v45 & 0xF) != 9 )
      v44[33] |= 0x40u;
    sub_BD6B90(v44, (unsigned __int8 *)*a2);
    sub_BD84D0(*a2, (__int64)v44);
    return sub_B30290(*a2);
  }
  return result;
}
