// Function: sub_24E76E0
// Address: 0x24e76e0
//
__int64 __fastcall sub_24E76E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v7; // bl
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned int *v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // r14
  _QWORD *v19; // rax
  __int64 v20; // r12
  __int64 v21; // rdx
  unsigned __int64 v22; // r14
  __int64 v23; // rdx
  unsigned int v24; // esi
  _QWORD *v25; // rdi
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  int v28; // edx
  _QWORD *v29; // rdi
  _QWORD *v30; // rax
  __int64 *v31; // rdi
  __int64 v32; // rsi
  _BYTE *v34; // [rsp+18h] [rbp-F8h]
  _BYTE v35[32]; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v36; // [rsp+40h] [rbp-D0h]
  _BYTE *v37; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+58h] [rbp-B8h]
  _BYTE v39[32]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+80h] [rbp-90h]
  __int64 v41; // [rsp+88h] [rbp-88h]
  __int64 v42; // [rsp+90h] [rbp-80h]
  __int64 v43; // [rsp+98h] [rbp-78h]
  void **v44; // [rsp+A0h] [rbp-70h]
  void **v45; // [rsp+A8h] [rbp-68h]
  __int64 v46; // [rsp+B0h] [rbp-60h]
  int v47; // [rsp+B8h] [rbp-58h]
  __int16 v48; // [rsp+BCh] [rbp-54h]
  char v49; // [rsp+BEh] [rbp-52h]
  __int64 v50; // [rsp+C0h] [rbp-50h]
  __int64 v51; // [rsp+C8h] [rbp-48h]
  void *v52; // [rsp+D0h] [rbp-40h] BYREF
  void *v53; // [rsp+D8h] [rbp-38h] BYREF

  v7 = a4;
  if ( !sub_AD7A80(
          *(_BYTE **)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))),
          a2,
          *(_DWORD *)(a1 + 4) & 0x7FFFFFF,
          a4,
          a5) )
  {
    sub_24E6B60((unsigned __int8 *)a1, a2, a3, v7);
    goto LABEL_28;
  }
  v45 = &v53;
  v43 = sub_BD5C60(a1);
  v44 = &v52;
  v37 = v39;
  v52 = &unk_49DA100;
  v38 = 0x200000000LL;
  LOWORD(v42) = 0;
  v53 = &unk_49DA0B0;
  v46 = 0;
  v47 = 0;
  v48 = 512;
  v49 = 7;
  v50 = 0;
  v51 = 0;
  v40 = 0;
  v41 = 0;
  sub_D5F1F0((__int64)&v37, a1);
  v8 = *(_DWORD *)(a2 + 280);
  if ( v8 )
  {
    if ( (unsigned int)(v8 - 1) <= 1 && !*(_BYTE *)(a2 + 360) )
      sub_24F4CC0(a2, &v37, a3, 0);
  }
  else
  {
    sub_24E58A0((__int64)&v37, a2, a3);
    if ( !v7 )
    {
      sub_F94A20(&v37, a2);
      v31 = (__int64 *)sub_BD5C60(a1);
      goto LABEL_26;
    }
  }
  if ( *(char *)(a1 + 7) < 0 )
  {
    v9 = sub_BD2BC0(a1);
    v11 = v9 + v10;
    if ( *(char *)(a1 + 7) < 0 )
      v11 -= sub_BD2BC0(a1);
    v12 = v11 >> 4;
    if ( (_DWORD)v12 )
    {
      v13 = 0;
      v14 = 16LL * (unsigned int)v12;
      while ( 1 )
      {
        v15 = 0;
        if ( *(char *)(a1 + 7) < 0 )
          v15 = sub_BD2BC0(a1);
        v16 = (unsigned int *)(v13 + v15);
        if ( *(_DWORD *)(*(_QWORD *)v16 + 8LL) == 1 )
          break;
        v13 += 16;
        if ( v14 == v13 )
          goto LABEL_24;
      }
      v17 = 32 * (v16[2] - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v36 = 257;
      v18 = *(_QWORD *)(a1 + v17);
      v19 = sub_BD2C40(72, 1u);
      v20 = (__int64)v19;
      if ( v19 )
        sub_B4BF70((__int64)v19, v18, 0, 1u, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v45 + 2))(v45, v20, v35, v41, v42);
      v21 = 16LL * (unsigned int)v38;
      v22 = (unsigned __int64)v37;
      v34 = &v37[v21];
      if ( v37 != &v37[v21] )
      {
        do
        {
          v23 = *(_QWORD *)(v22 + 8);
          v24 = *(_DWORD *)v22;
          v22 += 16LL;
          sub_B99FD0(v20, v24, v23);
        }
        while ( v34 != (_BYTE *)v22 );
      }
      v25 = *(_QWORD **)(a1 + 40);
      v36 = 257;
      sub_AA8550(v25, (__int64 *)(a1 + 24), 0, (__int64)v35, 0);
      v26 = *(_QWORD *)(v20 + 40);
      v27 = *(_QWORD *)(v26 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v27 == v26 + 48 )
      {
        v29 = 0;
      }
      else
      {
        if ( !v27 )
          BUG();
        v28 = *(unsigned __int8 *)(v27 - 24);
        v29 = 0;
        v30 = (_QWORD *)(v27 - 24);
        if ( (unsigned int)(v28 - 30) < 0xB )
          v29 = v30;
      }
      sub_B43D60(v29);
    }
  }
LABEL_24:
  nullsub_61();
  v52 = &unk_49DA100;
  nullsub_63();
  if ( v37 == v39 )
  {
LABEL_28:
    v31 = (__int64 *)sub_BD5C60(a1);
    if ( !v7 )
      goto LABEL_26;
LABEL_29:
    v32 = sub_ACD6D0(v31);
    goto LABEL_30;
  }
  _libc_free((unsigned __int64)v37);
  v31 = (__int64 *)sub_BD5C60(a1);
  if ( v7 )
    goto LABEL_29;
LABEL_26:
  v32 = sub_ACD720(v31);
LABEL_30:
  sub_BD84D0(a1, v32);
  return sub_B43D60((_QWORD *)a1);
}
