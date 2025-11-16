// Function: sub_2CD2F00
// Address: 0x2cd2f00
//
void __fastcall sub_2CD2F00(_BYTE *a1, __int64 a2, __int64 a3, unsigned __int64 a4, int a5)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  _QWORD *v8; // r12
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r9
  unsigned __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rdi
  char v20; // bl
  const void *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // r9
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  int v26; // edx
  __int64 v27; // rax
  unsigned __int64 v28; // rsi
  __int64 v29; // rax
  __int64 **v30; // rax
  unsigned __int8 v31; // [rsp+0h] [rbp-1A0h]
  unsigned __int8 v32; // [rsp+0h] [rbp-1A0h]
  __int64 v33; // [rsp+0h] [rbp-1A0h]
  __int64 v34; // [rsp+8h] [rbp-198h]
  __int64 v35; // [rsp+8h] [rbp-198h]
  unsigned __int64 v36; // [rsp+8h] [rbp-198h]
  __int64 v40; // [rsp+28h] [rbp-178h]
  __int64 v41; // [rsp+28h] [rbp-178h]
  unsigned __int64 v42; // [rsp+28h] [rbp-178h]
  int v43; // [rsp+48h] [rbp-158h]
  _BYTE *v44; // [rsp+50h] [rbp-150h] BYREF
  __int64 v45; // [rsp+58h] [rbp-148h]
  _BYTE v46[32]; // [rsp+60h] [rbp-140h] BYREF
  _BYTE *v47; // [rsp+80h] [rbp-120h] BYREF
  __int64 v48; // [rsp+88h] [rbp-118h]
  _BYTE v49[32]; // [rsp+90h] [rbp-110h] BYREF
  _BYTE v50[32]; // [rsp+B0h] [rbp-F0h] BYREF
  __int16 v51; // [rsp+D0h] [rbp-D0h]
  unsigned int *v52[2]; // [rsp+E0h] [rbp-C0h] BYREF
  char v53; // [rsp+F0h] [rbp-B0h] BYREF
  void *v54; // [rsp+160h] [rbp-40h]

  v6 = 8LL * a5;
  v7 = 0;
  v44 = v46;
  v47 = v49;
  v45 = 0x300000000LL;
  v48 = 0x300000000LL;
  v8 = (_QWORD *)sub_BD5C60(a2);
  sub_23D0AB0((__int64)v52, a2, 0, 0, 0);
  v9 = 0;
  do
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v10 = *(_QWORD *)(a2 - 8);
      else
        v10 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v11 = (unsigned int)v45;
      v12 = *(_QWORD *)(v10 + 4 * v7);
      v13 = (unsigned int)v45 + 1LL;
      if ( v13 > HIDWORD(v45) )
      {
        v32 = v9;
        v35 = v12;
        sub_C8D5F0((__int64)&v44, v46, v13, 8u, v9, v12);
        v11 = (unsigned int)v45;
        v9 = v32;
        v12 = v35;
      }
      *(_QWORD *)&v44[8 * v11] = v12;
      LODWORD(v45) = v45 + 1;
      v14 = *(_QWORD *)(*(_QWORD *)&v44[v7] + 8LL);
      v15 = (unsigned int)v48;
      v16 = (unsigned int)v48 + 1LL;
      if ( v16 > HIDWORD(v48) )
      {
        v31 = v9;
        v34 = *(_QWORD *)(*(_QWORD *)&v44[v7] + 8LL);
        sub_C8D5F0((__int64)&v47, v49, v16, 8u, v9, v14);
        v15 = (unsigned int)v48;
        v9 = v31;
        v14 = v34;
      }
      *(_QWORD *)&v47[8 * v15] = v14;
      LODWORD(v48) = v48 + 1;
      if ( *(_BYTE *)(*(_QWORD *)&v47[v7] + 8LL) == 5 )
        break;
      v7 += 8;
      if ( v6 == v7 )
        goto LABEL_12;
    }
    v17 = sub_BCB2F0(v8);
    *(_QWORD *)&v47[v7] = v17;
    v51 = 257;
    v18 = sub_2CD24F0((__int64 *)v52, 0x31u, *(_QWORD *)&v44[v7], *(__int64 ***)&v47[v7], (__int64)v50, 0, v43, 0);
    v9 = 1;
    *(_QWORD *)&v44[v7] = v18;
    v7 += 8;
  }
  while ( v6 != v7 );
LABEL_12:
  v19 = *(_QWORD *)(a2 + 8);
  v20 = *(_BYTE *)(v19 + 8);
  if ( v20 == 5 || (_BYTE)v9 )
  {
    v21 = v47;
    v22 = (unsigned int)v48;
    v23 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) + 40LL);
    if ( v20 == 5 )
    {
      v33 = (unsigned int)v48;
      v36 = (unsigned __int64)v47;
      v41 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) + 40LL);
      v29 = sub_BCB2F0(v8);
      v22 = v33;
      v21 = (const void *)v36;
      v23 = v41;
      v19 = v29;
    }
    v40 = v23;
    v24 = sub_BCF480((__int64 *)v19, v21, v22, 0);
    v25 = sub_BA8C10(v40, a3, a4, v24, 0);
    v51 = 257;
    v27 = sub_921880(v52, v25, v26, (int)v44, v45, (__int64)v50, 0);
    v28 = v27;
    if ( v20 == 5 )
    {
      v42 = v27;
      v51 = 257;
      v30 = (__int64 **)sub_BCB1B0(v8);
      v28 = sub_2CD24F0((__int64 *)v52, 0x31u, v42, v30, (__int64)v50, 0, v43, 0);
    }
    sub_BD84D0(a2, v28);
    sub_B43D60((_QWORD *)a2);
    *a1 = 1;
  }
  nullsub_61();
  v54 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v52[0] != &v53 )
    _libc_free((unsigned __int64)v52[0]);
  if ( v47 != v49 )
    _libc_free((unsigned __int64)v47);
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
}
