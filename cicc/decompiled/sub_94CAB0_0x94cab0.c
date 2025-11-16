// Function: sub_94CAB0
// Address: 0x94cab0
//
__int64 __fastcall sub_94CAB0(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v4; // rax
  unsigned int v6; // r15d
  unsigned int v7; // r9d
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r12
  __m128i *v11; // rbx
  __m128i *v12; // r14
  __m128i *v13; // rax
  __int64 *v14; // rdi
  __m128i *v15; // r12
  __int64 v16; // rax
  __int64 v17; // r12
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // r12
  int v22; // eax
  __int64 v23; // rdi
  int v24; // r15d
  __int64 v25; // rax
  char v26; // al
  __int16 v27; // bx
  __int64 v28; // rax
  __int64 v29; // rbx
  unsigned int *v30; // r15
  unsigned int *v31; // r12
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  int v36; // eax
  unsigned __int64 v37; // rsi
  _QWORD *v38; // rdi
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v44; // [rsp+18h] [rbp-168h]
  __m128i *v45; // [rsp+28h] [rbp-158h]
  __int64 *v46; // [rsp+30h] [rbp-150h]
  int v47; // [rsp+40h] [rbp-140h]
  __int16 v48; // [rsp+46h] [rbp-13Ah]
  unsigned int v49; // [rsp+48h] [rbp-138h]
  unsigned int v50; // [rsp+48h] [rbp-138h]
  __int64 v51; // [rsp+58h] [rbp-128h] BYREF
  _BYTE v52[32]; // [rsp+60h] [rbp-120h] BYREF
  __int16 v53; // [rsp+80h] [rbp-100h]
  _BYTE v54[32]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v55; // [rsp+B0h] [rbp-D0h]
  _QWORD *v56; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+C8h] [rbp-B8h]
  _QWORD v58[22]; // [rsp+D0h] [rbp-B0h] BYREF

  v4 = (unsigned int)(a3 - 678);
  if ( (unsigned int)v4 > 0x1D )
  {
    v41 = (unsigned int)(a3 - 708);
    if ( (unsigned int)v41 > 0x17 )
    {
      v42 = (unsigned int)(a3 - 732);
      if ( (unsigned int)v42 > 0xC )
        sub_91B980("unexpected WMMA intrinsic!", 0);
      v6 = dword_3F147A0[v42];
      v7 = v6 - 8838;
    }
    else
    {
      v6 = dword_3F147E0[v41];
      v7 = v6 - 8838;
    }
  }
  else
  {
    v6 = dword_3F14840[v4];
    v7 = v6 - 8838;
  }
  v49 = v7;
  v8 = *(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL);
  v9 = *(_QWORD *)(*(_QWORD *)(v8 + 16) + 16LL);
  v46 = *(__int64 **)(v8 + 16);
  v10 = *(_QWORD *)(v9 + 16);
  sub_9480A0(v10, 1u, "unexpected 'rowcol' operand", "'rowcol' operand can be 0 or 1 only", (_DWORD *)(a4 + 36));
  v11 = sub_92F410(a2, v8);
  v45 = sub_92F410(a2, (__int64)v46);
  v12 = sub_92F410(a2, v9);
  v13 = sub_92F410(a2, v10);
  v14 = *(__int64 **)(a2 + 32);
  v15 = v13;
  v51 = v11->m128i_i64[1];
  v16 = sub_90A810(v14, v6, (__int64)&v51, 1u);
  v58[0] = v11;
  v44 = v16;
  v56 = v58;
  v58[1] = v12;
  v58[2] = v15;
  v57 = 0x1000000003LL;
  if ( v49 > 0x34 || (v40 = 0x10000001004001LL, v47 = 8, !_bittest64(&v40, v49)) )
  {
    v47 = 8;
    if ( (v6 & 0xFFFFFFF7) != 0x22C2 )
    {
      if ( v6 == 8914 || (v47 = 4, v6 == 8280) )
        v47 = 2;
    }
  }
  v50 = 0;
  do
  {
    v17 = *(_QWORD *)(a2 + 32) + 8LL;
    v18 = sub_8D46C0(*v46);
    v20 = sub_91A390(v17, v18, 0, v19);
    v55 = 257;
    v21 = v20;
    v22 = sub_94B2B0((unsigned int **)(a2 + 48), v20, (__int64)v45, v50, (__int64)v54);
    v23 = *(_QWORD *)(a2 + 96);
    v53 = 257;
    v24 = v22;
    v25 = sub_AA4E30(v23);
    v26 = sub_AE5020(v25, v21);
    HIBYTE(v27) = HIBYTE(v48);
    v55 = 257;
    LOBYTE(v27) = v26;
    v48 = v27;
    v28 = sub_BD2C40(80, unk_3F10A14);
    v29 = v28;
    if ( v28 )
      sub_B4D190(v28, v21, v24, (unsigned int)v54, 0, (unsigned __int8)v48, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v29,
      v52,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112));
    v30 = *(unsigned int **)(a2 + 48);
    v31 = &v30[4 * *(unsigned int *)(a2 + 56)];
    while ( v31 != v30 )
    {
      v32 = *((_QWORD *)v30 + 1);
      v33 = *v30;
      v30 += 4;
      sub_B99FD0(v29, v33, v32);
    }
    v34 = (unsigned int)v57;
    v35 = (unsigned int)v57 + 1LL;
    if ( v35 > HIDWORD(v57) )
    {
      sub_C8D5F0(&v56, v58, v35, 8);
      v34 = (unsigned int)v57;
    }
    ++v50;
    v56[v34] = v29;
    v36 = v57 + 1;
    LODWORD(v57) = v57 + 1;
  }
  while ( v47 != v50 );
  v37 = 0;
  v55 = 257;
  if ( v44 )
    v37 = *(_QWORD *)(v44 + 24);
  sub_921880((unsigned int **)(a2 + 48), v37, v44, (int)v56, v36, (__int64)v54, 0);
  v38 = v56;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v38 != v58 )
    _libc_free(v38, v37);
  return a1;
}
