// Function: sub_2C1C770
// Address: 0x2c1c770
//
void __fastcall sub_2C1C770(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 *v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // r15
  bool (__fastcall *v9)(__int64, __int64); // rax
  bool v10; // al
  char v11; // dl
  __int64 v12; // r15
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r10
  unsigned __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // rax
  _BYTE *v22; // r12
  __int64 v23; // r13
  __int64 v24; // rdi
  unsigned __int64 *v25; // r11
  __int64 v26; // r9
  unsigned int **v27; // rdi
  unsigned __int64 v28; // rsi
  __int64 v29; // r13
  __int64 v30; // rbx
  unsigned __int64 *v31; // r13
  unsigned __int64 v32; // rdi
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r12
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // [rsp+0h] [rbp-130h]
  __int64 *i; // [rsp+28h] [rbp-108h]
  _BYTE *v40; // [rsp+30h] [rbp-100h] BYREF
  __int64 v41; // [rsp+38h] [rbp-F8h]
  _BYTE v42[16]; // [rsp+40h] [rbp-F0h] BYREF
  char v43[32]; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v44; // [rsp+70h] [rbp-C0h]
  _BYTE *v45; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+88h] [rbp-A8h]
  _BYTE v47[32]; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 *v48; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v49; // [rsp+B8h] [rbp-78h]
  _BYTE v50[112]; // [rsp+C0h] [rbp-70h] BYREF

  v48 = *(unsigned __int64 **)(a1 + 88);
  if ( v48 )
    sub_2AAAFA0((__int64 *)&v48);
  sub_2BF1A90((__int64)a2, (__int64)&v48);
  sub_9C6650(&v48);
  v4 = *(unsigned int *)(a1 + 160);
  v5 = *a2;
  v40 = v42;
  v41 = 0x200000000LL;
  if ( sub_9B76D0(v4, 0xFFFFFFFF, v5) )
  {
    v35 = sub_BCE1B0(*(__int64 **)(a1 + 168), a2[1]);
    v36 = (unsigned int)v41;
    v37 = (unsigned int)v41 + 1LL;
    if ( v37 > HIDWORD(v41) )
    {
      sub_C8D5F0((__int64)&v40, v42, v37, 8u, v33, v34);
      v36 = (unsigned int)v41;
    }
    *(_QWORD *)&v40[8 * v36] = v35;
    LODWORD(v41) = v41 + 1;
  }
  v6 = *(__int64 **)(a1 + 48);
  v7 = 0;
  v45 = v47;
  v46 = 0x400000000LL;
  for ( i = &v6[*(unsigned int *)(a1 + 56)]; i != v6; LODWORD(v46) = v46 + 1 )
  {
    if ( sub_9B75A0(*(unsigned int *)(a1 + 160), v7, *a2) )
    {
      BYTE4(v48) = 0;
      LODWORD(v48) = 0;
      v12 = sub_2BFB120((__int64)a2, *v6, (unsigned int *)&v48);
    }
    else
    {
      v8 = *v6;
      v9 = *(bool (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL);
      if ( v9 == sub_2C0D510 )
      {
        v10 = sub_B5A760(*(_DWORD *)(a1 + 160));
        v11 = 0;
        if ( v10 )
          v11 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1)) == v8;
      }
      else
      {
        v11 = v9(a1, *v6);
      }
      v12 = sub_2BFB640((__int64)a2, *v6, v11);
    }
    if ( sub_9B76D0(*(unsigned int *)(a1 + 160), v7, *a2) )
    {
      v17 = (unsigned int)v41;
      v18 = *(_QWORD *)(v12 + 8);
      v19 = (unsigned int)v41 + 1LL;
      if ( v19 > HIDWORD(v41) )
      {
        v38 = *(_QWORD *)(v12 + 8);
        sub_C8D5F0((__int64)&v40, v42, v19, 8u, v13, v14);
        v17 = (unsigned int)v41;
        v18 = v38;
      }
      *(_QWORD *)&v40[8 * v17] = v18;
      LODWORD(v41) = v41 + 1;
    }
    v15 = (unsigned int)v46;
    v16 = (unsigned int)v46 + 1LL;
    if ( v16 > HIDWORD(v46) )
    {
      sub_C8D5F0((__int64)&v45, v47, v16, 8u, v13, v14);
      v15 = (unsigned int)v46;
    }
    ++v7;
    ++v6;
    *(_QWORD *)&v45[8 * v15] = v12;
  }
  v20 = (__int64 *)sub_AA4B30(*(_QWORD *)(a2[113] + 48));
  v21 = sub_B6E160(v20, *(_DWORD *)(a1 + 160), (__int64)v40, (unsigned int)v41);
  v22 = *(_BYTE **)(a1 + 136);
  v23 = v21;
  if ( v22 )
  {
    v24 = *(_QWORD *)(a1 + 136);
    v48 = (unsigned __int64 *)v50;
    v49 = 0x100000000LL;
    sub_B56970(v24, (__int64)&v48);
    v25 = v48;
    v26 = (unsigned int)v49;
  }
  else
  {
    v26 = 0;
    v48 = (unsigned __int64 *)v50;
    v25 = (unsigned __int64 *)v50;
    v49 = 0x100000000LL;
  }
  v27 = (unsigned int **)a2[113];
  v28 = 0;
  v44 = 257;
  if ( v23 )
    v28 = *(_QWORD *)(v23 + 24);
  v29 = sub_B33530(v27, v28, v23, (int)v45, v46, (__int64)v43, (__int64)v25, v26, 0);
  sub_2AAF930(a1, (unsigned __int8 *)v29);
  if ( *(_BYTE *)(*(_QWORD *)(v29 + 8) + 8LL) != 7 )
    sub_2BF26E0((__int64)a2, a1 + 96, v29, 0);
  sub_2BF08A0((__int64)a2, (_BYTE *)v29, v22);
  v30 = (__int64)v48;
  v31 = &v48[7 * (unsigned int)v49];
  if ( v48 != v31 )
  {
    do
    {
      v32 = *(v31 - 3);
      v31 -= 7;
      if ( v32 )
        j_j___libc_free_0(v32);
      if ( (unsigned __int64 *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31);
    }
    while ( (unsigned __int64 *)v30 != v31 );
    v31 = v48;
  }
  if ( v31 != (unsigned __int64 *)v50 )
    _libc_free((unsigned __int64)v31);
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
}
