// Function: sub_24165D0
// Address: 0x24165d0
//
void __fastcall sub_24165D0(__int64 *a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  unsigned __int8 *v6; // r13
  int v7; // eax
  __int64 *v8; // r14
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  unsigned int v23; // r12d
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  int v27; // edx
  int v28; // eax
  __int64 i; // r12
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rbx
  int v33; // ebx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  char v41; // al
  __int16 v42; // cx
  _QWORD *v43; // rax
  __int64 v44; // r9
  __int64 v45; // r12
  unsigned int *v46; // r15
  __int64 v47; // rbx
  __int64 v48; // rdx
  unsigned int v49; // esi
  int v50; // edx
  unsigned int v51; // r13d
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // r14
  _QWORD *v55; // rax
  _QWORD *v56; // r12
  __int64 v57; // [rsp-8h] [rbp-C8h]
  __int64 v58; // [rsp+0h] [rbp-C0h]
  _QWORD *v59; // [rsp+8h] [rbp-B8h]
  __int64 *v61; // [rsp+10h] [rbp-B0h]
  __int64 v62; // [rsp+18h] [rbp-A8h]
  __int64 v64; // [rsp+20h] [rbp-A0h]
  __int64 v65; // [rsp+28h] [rbp-98h]
  __int16 v66; // [rsp+36h] [rbp-8Ah]
  __int64 v67; // [rsp+40h] [rbp-80h]
  unsigned int v68; // [rsp+40h] [rbp-80h]
  __int64 *v69; // [rsp+48h] [rbp-78h]
  __int64 v70; // [rsp+58h] [rbp-68h] BYREF
  _QWORD v71[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v72; // [rsp+80h] [rbp-40h]

  v5 = a5;
  v6 = a3;
  v58 = a4;
  v62 = *(_QWORD *)(a2 + 24);
  v7 = *(_DWORD *)(v62 + 12);
  if ( v7 == 1 )
  {
    v69 = (__int64 *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
  }
  else
  {
    v8 = (__int64 *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v10 = (__int64)(a3 + 24);
    v69 = &v8[4 * (unsigned int)(v7 - 1)];
    do
    {
      v11 = *v8;
      v8 += 4;
      v67 = *a1;
      v12 = sub_24159D0(*a1, v11);
      v71[0] = sub_2415280(v67, v12, v10, 0);
      sub_240DEA0(a4, v71);
    }
    while ( v8 != v69 );
    v6 = a3;
    v5 = a5;
  }
  if ( !(*(_DWORD *)(v62 + 8) >> 8) )
    goto LABEL_6;
  v14 = *v6;
  if ( v14 == 40 )
  {
    v15 = 32LL * (unsigned int)sub_B491D0((__int64)v6);
  }
  else
  {
    v15 = 0;
    if ( v14 != 85 )
    {
      v15 = 64;
      if ( v14 != 34 )
        goto LABEL_52;
    }
  }
  if ( (v6[7] & 0x80u) == 0 )
    goto LABEL_21;
  v16 = sub_BD2BC0((__int64)v6);
  v18 = v16 + v17;
  if ( (v6[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v18 >> 4) )
LABEL_51:
      BUG();
LABEL_21:
    v22 = 0;
    goto LABEL_22;
  }
  if ( !(unsigned int)((v18 - sub_BD2BC0((__int64)v6)) >> 4) )
    goto LABEL_21;
  if ( (v6[7] & 0x80u) == 0 )
    goto LABEL_51;
  v19 = *(_DWORD *)(sub_BD2BC0((__int64)v6) + 8);
  if ( (v6[7] & 0x80u) == 0 )
    BUG();
  v20 = sub_BD2BC0((__int64)v6);
  v22 = 32LL * (unsigned int)(*(_DWORD *)(v20 + v21 - 4) - v19);
LABEL_22:
  v61 = sub_BCD420(
          *(__int64 **)(*(_QWORD *)*a1 + 48LL),
          (unsigned int)((32LL * (*((_DWORD *)v6 + 1) & 0x7FFFFFF) - 32 - v15 - v22) >> 5) - *(_DWORD *)(v62 + 12) + 1);
  v23 = *(_DWORD *)(sub_B2BEC0(*(_QWORD *)(*a1 + 8)) + 4);
  v71[0] = "labelva";
  v24 = *a1;
  v72 = 259;
  v25 = *(_QWORD *)(*(_QWORD *)(v24 + 8) + 80LL);
  if ( !v25 )
    BUG();
  v26 = *(_QWORD *)(v25 + 32);
  v59 = sub_BD2C40(80, unk_3F10A14);
  if ( v59 )
    sub_B4CE50((__int64)v59, v61, v23, (__int64)v71, v26, 1);
  v27 = *v6;
  v68 = 0;
  v28 = v27 - 29;
  if ( v27 == 40 )
    goto LABEL_41;
LABEL_26:
  i = -32;
  if ( v28 == 56 )
    goto LABEL_29;
  if ( v28 != 5 )
LABEL_52:
    BUG();
  for ( i = -96; ; i = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v6) )
  {
LABEL_29:
    if ( (v6[7] & 0x80u) != 0 )
    {
      v30 = sub_BD2BC0((__int64)v6);
      v32 = v30 + v31;
      if ( (v6[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v32 >> 4) )
LABEL_53:
          BUG();
      }
      else if ( (unsigned int)((v32 - sub_BD2BC0((__int64)v6)) >> 4) )
      {
        if ( (v6[7] & 0x80u) == 0 )
          goto LABEL_53;
        v33 = *(_DWORD *)(sub_BD2BC0((__int64)v6) + 8);
        if ( (v6[7] & 0x80u) == 0 )
          BUG();
        v34 = sub_BD2BC0((__int64)v6);
        i -= 32LL * (unsigned int)(*(_DWORD *)(v34 + v35 - 4) - v33);
      }
    }
    v72 = 257;
    if ( v69 == (__int64 *)&v6[i] )
      break;
    v64 = sub_9213A0((unsigned int **)v5, (__int64)v61, (__int64)v59, 0, v68, (__int64)v71, 7u);
    v36 = v65;
    v37 = *a1;
    LOWORD(v36) = 0;
    v65 = v36;
    v38 = sub_24159D0(*a1, *v69);
    v39 = sub_2415280(v37, v38, (__int64)(v6 + 24), v65);
    v40 = sub_AA4E30(*(_QWORD *)(v5 + 48));
    v41 = sub_AE5020(v40, *(_QWORD *)(v39 + 8));
    HIBYTE(v42) = HIBYTE(v66);
    v72 = 257;
    LOBYTE(v42) = v41;
    v66 = v42;
    v43 = sub_BD2C40(80, unk_3F10A10);
    v44 = v57;
    v45 = (__int64)v43;
    if ( v43 )
      sub_B4D3C0((__int64)v43, v39, v64, 0, v66, v57, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v5 + 88) + 16LL))(
      *(_QWORD *)(v5 + 88),
      v45,
      v71,
      *(_QWORD *)(v5 + 56),
      *(_QWORD *)(v5 + 64),
      v44,
      v58);
    v46 = *(unsigned int **)v5;
    v47 = *(_QWORD *)v5 + 16LL * *(unsigned int *)(v5 + 8);
    if ( *(_QWORD *)v5 != v47 )
    {
      do
      {
        v48 = *((_QWORD *)v46 + 1);
        v49 = *v46;
        v46 += 4;
        sub_B99FD0(v45, v49, v48);
      }
      while ( (unsigned int *)v47 != v46 );
    }
    v50 = *v6;
    v69 += 4;
    ++v68;
    v28 = v50 - 29;
    if ( v50 != 40 )
      goto LABEL_26;
LABEL_41:
    ;
  }
  v70 = sub_9213A0((unsigned int **)v5, (__int64)v61, (__int64)v59, 0, 0, (__int64)v71, 7u);
  sub_240DEA0(v58, &v70);
LABEL_6:
  if ( *(_BYTE *)(**(_QWORD **)(v62 + 16) + 8LL) != 7 )
  {
    v13 = *(_QWORD *)(*a1 + 160);
    if ( !v13 )
    {
      v51 = *(_DWORD *)(sub_B2BEC0(*(_QWORD *)(*a1 + 8)) + 4);
      v71[0] = "labelreturn";
      v52 = *a1;
      v72 = 259;
      v53 = *(_QWORD *)(*(_QWORD *)(v52 + 8) + 80LL);
      if ( !v53 )
        BUG();
      v54 = *(_QWORD *)(v53 + 32);
      v55 = sub_BD2C40(80, unk_3F10A14);
      v56 = v55;
      if ( v55 )
        sub_B4CE50((__int64)v55, *(__int64 **)(*(_QWORD *)*a1 + 48LL), v51, (__int64)v71, v54, 1);
      *(_QWORD *)(*a1 + 160) = v56;
      v13 = *(_QWORD *)(*a1 + 160);
    }
    v71[0] = v13;
    sub_240DEA0(v58, v71);
  }
}
