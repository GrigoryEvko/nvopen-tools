// Function: sub_F88DC0
// Address: 0xf88dc0
//
__int64 __fastcall sub_F88DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r14
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int8 *v18; // r15
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int16 v21; // ax
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // r15
  __int64 v26; // r13
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // rbx
  __int64 v30; // r13
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // r12
  __int64 v34; // r14
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rax
  __int16 v38; // ax
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  unsigned __int64 v42; // rcx
  __int64 v43; // rax
  __int16 v44; // dx
  __int64 v45; // r15
  char v46; // [rsp+Fh] [rbp-C1h]
  __int64 v47; // [rsp+10h] [rbp-C0h]
  char v48; // [rsp+27h] [rbp-A9h] BYREF
  __int64 v49; // [rsp+28h] [rbp-A8h] BYREF
  _BYTE v50[32]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v51; // [rsp+50h] [rbp-80h]
  __int64 v52; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v53; // [rsp+68h] [rbp-68h] BYREF
  __int64 v54; // [rsp+70h] [rbp-60h]
  __int64 v55; // [rsp+78h] [rbp-58h]
  __int64 v56; // [rsp+80h] [rbp-50h] BYREF
  __int16 v57; // [rsp+88h] [rbp-48h]
  _QWORD v58[8]; // [rsp+90h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 48);
  if ( *(_BYTE *)(a1 + 444) )
  {
    v7 = *(_QWORD **)(a1 + 424);
    v8 = (__int64)&v7[*(unsigned int *)(a1 + 436)];
    if ( v7 == (_QWORD *)v8 )
      goto LABEL_9;
    while ( v6 != *v7 )
    {
      if ( (_QWORD *)v8 == ++v7 )
        goto LABEL_9;
    }
  }
  else if ( !sub_C8CA60(a1 + 416, v6) )
  {
LABEL_9:
    v10 = a2;
    goto LABEL_10;
  }
  v9 = *(_QWORD *)a1;
  v53 = &v56;
  v54 = 0x100000002LL;
  LODWORD(v55) = 0;
  BYTE4(v55) = 1;
  v56 = v6;
  v52 = 1;
  v10 = sub_1055B50(a2, &v52, v9, 0);
  if ( !BYTE4(v55) )
    _libc_free(v53, &v52);
LABEL_10:
  v47 = sub_D33D80((_QWORD *)v10, *(_QWORD *)a1, v8, a4, a5);
  v49 = 0;
  v48 = 0;
  v11 = sub_F879B0(a1, v10, v6, &v49, &v48);
  if ( *(_BYTE *)(a1 + 444) )
  {
    v12 = *(_QWORD **)(a1 + 424);
    v13 = &v12[*(unsigned int *)(a1 + 436)];
    if ( v12 == v13 )
      goto LABEL_32;
    while ( v6 != *v12 )
    {
      if ( v13 == ++v12 )
        goto LABEL_32;
    }
  }
  else if ( !sub_C8CA60(a1 + 416, v6) )
  {
    goto LABEL_32;
  }
  v14 = sub_D47930(v6);
  if ( (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != 0 )
  {
    v15 = *(_QWORD *)(v11 - 8);
    v16 = 0;
    do
    {
      if ( v14 == *(_QWORD *)(v15 + 32LL * *(unsigned int *)(v11 + 72) + 8 * v16) )
      {
        v17 = 32 * v16;
        goto LABEL_20;
      }
      ++v16;
    }
    while ( (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != (_DWORD)v16 );
    v17 = 0x1FFFFFFFE0LL;
  }
  else
  {
    v17 = 0x1FFFFFFFE0LL;
    v15 = *(_QWORD *)(v11 - 8);
  }
LABEL_20:
  v18 = *(unsigned __int8 **)(v15 + v17);
  v19 = *v18;
  if ( (unsigned __int8)v19 <= 0x1Cu )
  {
    if ( (_BYTE)v19 != 5 || (*((_WORD *)v18 + 1) & 0xFFFD) != 0xD && (*((_WORD *)v18 + 1) & 0xFFF7) != 0x11 )
    {
LABEL_31:
      v11 = (__int64)v18;
      goto LABEL_32;
    }
  }
  else
  {
    if ( (unsigned __int8)v19 > 0x36u )
      goto LABEL_28;
    v20 = 0x40540000000000LL;
    if ( !_bittest64(&v20, v19) )
      goto LABEL_28;
  }
  v21 = *(_WORD *)(a2 + 28);
  if ( (v21 & 2) == 0 )
  {
    sub_B447F0(v18, 0);
    v21 = *(_WORD *)(a2 + 28);
  }
  if ( (v21 & 4) == 0 )
    sub_B44850(v18, 0);
  if ( *v18 <= 0x1Cu )
    goto LABEL_31;
LABEL_28:
  v22 = *(_QWORD *)(a1 + 576);
  if ( v22 )
    v22 -= 24;
  v46 = sub_B19DB0(*(_QWORD *)(*(_QWORD *)a1 + 40LL), (__int64)v18, v22);
  if ( v46 )
    goto LABEL_31;
  if ( *(_BYTE *)(sub_D95540(**(_QWORD **)(a2 + 32)) + 8) != 14 )
  {
    v46 = sub_D969D0(v47);
    if ( v46 )
      v47 = (__int64)sub_DCAF50(*(__int64 **)a1, v47, 0);
  }
  v37 = *(_QWORD *)(a1 + 568);
  v53 = 0;
  v52 = a1 + 520;
  v54 = 0;
  v55 = v37;
  if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
    sub_BD73F0((__int64)&v53);
  v38 = *(_WORD *)(a1 + 584);
  v56 = *(_QWORD *)(a1 + 576);
  v57 = v38;
  sub_B33910(v58, (__int64 *)(a1 + 520));
  v41 = *(unsigned int *)(a1 + 792);
  v42 = *(unsigned int *)(a1 + 796);
  v58[1] = a1;
  if ( v41 + 1 > v42 )
  {
    sub_C8D5F0(a1 + 784, (const void *)(a1 + 800), v41 + 1, 8u, v39, v40);
    v41 = *(unsigned int *)(a1 + 792);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 784) + 8 * v41) = &v52;
  ++*(_DWORD *)(a1 + 792);
  v43 = sub_AA5190(**(_QWORD **)(v6 + 32));
  if ( !v43 )
    BUG();
  sub_A88F30(a1 + 520, *(_QWORD *)(v43 + 16), v43, v44);
  v45 = sub_F894B0(a1, v47);
  sub_F80960((__int64)&v52);
  v11 = (__int64)sub_F7DD30(a1, v11, v45, v6, v46);
LABEL_32:
  v23 = v49;
  if ( v49 )
  {
    if ( v49 != *(_QWORD *)(v11 + 8) )
    {
      v51 = 257;
      if ( v49 == *(_QWORD *)(v11 + 8) )
      {
        v24 = v11;
      }
      else
      {
        v24 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 600) + 120LL))(
                *(_QWORD *)(a1 + 600),
                38,
                v11,
                v49);
        if ( !v24 )
        {
          LOWORD(v56) = 257;
          v24 = sub_B51D30(38, v11, v23, (__int64)&v52, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
            *(_QWORD *)(a1 + 608),
            v24,
            v50,
            *(_QWORD *)(a1 + 576),
            *(_QWORD *)(a1 + 584));
          v33 = *(_QWORD *)(a1 + 520);
          v34 = v33 + 16LL * *(unsigned int *)(a1 + 528);
          while ( v34 != v33 )
          {
            v35 = *(_QWORD *)(v33 + 8);
            v36 = *(_DWORD *)v33;
            v33 += 16;
            sub_B99FD0(v24, v36, v35);
          }
        }
      }
      v11 = v24;
    }
    if ( v48 )
    {
      v51 = 257;
      v26 = sub_F894B0(a1, **(_QWORD **)(v10 + 32));
      v27 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 600) + 32LL))(
              *(_QWORD *)(a1 + 600),
              15,
              v26,
              v11,
              0,
              0);
      if ( !v27 )
      {
        LOWORD(v56) = 257;
        v27 = sub_B504D0(15, v26, v11, (__int64)&v52, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
          *(_QWORD *)(a1 + 608),
          v27,
          v50,
          *(_QWORD *)(a1 + 576),
          *(_QWORD *)(a1 + 584));
        v28 = 16LL * *(unsigned int *)(a1 + 528);
        v29 = *(_QWORD *)(a1 + 520);
        v30 = v29 + v28;
        while ( v30 != v29 )
        {
          v31 = *(_QWORD *)(v29 + 8);
          v32 = *(_DWORD *)v29;
          v29 += 16;
          sub_B99FD0(v27, v32, v31);
        }
      }
      return v27;
    }
  }
  return v11;
}
