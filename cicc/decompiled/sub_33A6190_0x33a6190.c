// Function: sub_33A6190
// Address: 0x33a6190
//
void __fastcall sub_33A6190(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // eax
  __int64 v10; // rdi
  __int64 v11; // rcx
  int v12; // eax
  __int128 v13; // rax
  __int128 v14; // rax
  __int128 v15; // rax
  int v16; // r9d
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 (*v19)(); // rdx
  unsigned __int16 v20; // ax
  __int64 v21; // r12
  unsigned int v22; // edx
  unsigned __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rsi
  int v29; // edx
  __int128 v30; // rax
  int v31; // r9d
  __int64 v32; // rbx
  int v33; // edx
  _QWORD *v34; // rax
  __int64 v35; // rsi
  __int128 v36; // [rsp-20h] [rbp-E0h]
  __int64 *v37; // [rsp+8h] [rbp-B8h]
  __int128 v38; // [rsp+10h] [rbp-B0h]
  __int128 v39; // [rsp+20h] [rbp-A0h]
  __int128 v40; // [rsp+30h] [rbp-90h]
  unsigned int v41; // [rsp+40h] [rbp-80h]
  __int64 v42; // [rsp+40h] [rbp-80h]
  __int64 v43; // [rsp+78h] [rbp-48h] BYREF
  __int64 v44; // [rsp+80h] [rbp-40h] BYREF
  int v45; // [rsp+88h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 864);
  v5 = *(_DWORD *)(a1 + 848);
  v44 = 0;
  v6 = *(_QWORD *)(v4 + 16);
  v7 = *(_QWORD *)a1;
  v45 = v5;
  if ( v7 )
  {
    if ( &v44 != (__int64 *)(v7 + 48) )
    {
      v8 = *(_QWORD *)(v7 + 48);
      v44 = v8;
      if ( v8 )
        sub_B96E90((__int64)&v44, v8, 1);
    }
  }
  sub_B5B080(a2);
  v10 = v9;
  v11 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
  v12 = *(unsigned __int8 *)(v11 + 8);
  if ( (unsigned int)(v12 - 17) <= 1 )
    LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
  if ( (unsigned __int8)v12 <= 3u || (_BYTE)v12 == 5 || (v12 & 0xFD) == 4 )
  {
    v41 = sub_34B9180(v10);
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 856) + 864LL) & 4) != 0 )
      v41 = sub_34B9190(v41);
  }
  else
  {
    v41 = sub_34B9220(v10);
  }
  *(_QWORD *)&v13 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v40 = v13;
  *(_QWORD *)&v14 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v39 = v14;
  *(_QWORD *)&v15 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v38 = v15;
  sub_338B750(a1, *(_QWORD *)(a2 + 32 * (4LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v18 = v17;
  v19 = *(__int64 (**)())(*(_QWORD *)v6 + 80LL);
  v20 = 7;
  if ( v19 != sub_2FE2E20 )
    v20 = ((__int64 (__fastcall *)(__int64))v19)(v6);
  v21 = sub_33FAF80(*(_QWORD *)(a1 + 864), 214, (unsigned int)&v44, v20, 0, v16);
  v23 = v22 | v18 & 0xFFFFFFFF00000000LL;
  v24 = *(_QWORD *)(a1 + 864);
  v37 = *(__int64 **)(a2 + 8);
  v25 = *(_QWORD *)(v24 + 16);
  v26 = sub_2E79000(*(__int64 **)(v24 + 40));
  v27 = sub_2D5BAE0(v25, v26, v37, 0);
  v28 = v41;
  LODWORD(v25) = v29;
  LODWORD(v37) = v27;
  v42 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v30 = sub_33ED040(v42, v28);
  *((_QWORD *)&v36 + 1) = v23;
  *(_QWORD *)&v36 = v21;
  v32 = sub_33FC1D0(v42, 463, (unsigned int)&v44, (_DWORD)v37, v25, v31, v40, v39, v30, v38, v36);
  LODWORD(v21) = v33;
  v43 = a2;
  v34 = sub_337DC20(a1 + 8, &v43);
  *v34 = v32;
  v35 = v44;
  *((_DWORD *)v34 + 2) = v21;
  if ( v35 )
    sub_B91220((__int64)&v44, v35);
}
