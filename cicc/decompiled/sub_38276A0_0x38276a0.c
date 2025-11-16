// Function: sub_38276A0
// Address: 0x38276a0
//
void __fastcall sub_38276A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  unsigned __int16 *v6; // rdx
  __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 v9; // rdx
  __int64 v10; // rdx
  char v11; // al
  unsigned __int64 v12; // rax
  unsigned __int16 v13; // ax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  char v18; // al
  unsigned int v19; // eax
  __int64 v20; // r10
  __int64 v21; // rdx
  int v22; // r9d
  __int64 v23; // r11
  unsigned __int8 *v24; // r10
  unsigned int v25; // edx
  __int64 v26; // r9
  unsigned __int8 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int128 v31; // [rsp-30h] [rbp-100h]
  __int64 v32; // [rsp+8h] [rbp-C8h]
  __int64 v33; // [rsp+18h] [rbp-B8h]
  __int64 v34; // [rsp+18h] [rbp-B8h]
  unsigned int v37; // [rsp+40h] [rbp-90h] BYREF
  __int64 v38; // [rsp+48h] [rbp-88h]
  __int64 v39; // [rsp+50h] [rbp-80h] BYREF
  int v40; // [rsp+58h] [rbp-78h]
  unsigned __int64 v41; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-68h]
  unsigned __int64 v43; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-58h]
  __int64 v45; // [rsp+80h] [rbp-50h]
  __int64 v46; // [rsp+88h] [rbp-48h]
  __int64 v47; // [rsp+90h] [rbp-40h] BYREF
  __int64 v48; // [rsp+98h] [rbp-38h]

  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *((_QWORD *)v6 + 1);
  v8 = *v6;
  LOWORD(v47) = *v6;
  v48 = v7;
  if ( (_WORD)v47 )
  {
    if ( (_WORD)v47 == 1 || (unsigned __int16)(v47 - 504) <= 7u )
      goto LABEL_45;
    v29 = 16LL * ((unsigned __int16)v47 - 1);
    v10 = *(_QWORD *)&byte_444C4A0[v29];
    v11 = byte_444C4A0[v29 + 8];
  }
  else
  {
    v45 = sub_3007260((__int64)&v47);
    v46 = v9;
    v10 = v45;
    v11 = v46;
  }
  v47 = v10;
  LOBYTE(v48) = v11;
  v12 = (unsigned __int64)sub_CA1930(&v47) >> 1;
  switch ( (_DWORD)v12 )
  {
    case 1:
      v13 = 2;
      break;
    case 2:
      v13 = 3;
      break;
    case 4:
      v13 = 4;
      break;
    case 8:
      v13 = 5;
      break;
    case 0x10:
      v13 = 6;
      break;
    case 0x20:
      v13 = 7;
      break;
    case 0x40:
      v13 = 8;
      break;
    case 0x80:
      v13 = 9;
      break;
    default:
      v13 = sub_3007020(*(_QWORD **)(a1[1] + 64), v12);
      goto LABEL_14;
  }
  v14 = 0;
LABEL_14:
  v15 = *(_QWORD *)(a2 + 80);
  LOWORD(v37) = v13;
  v38 = v14;
  v39 = v15;
  if ( v15 )
  {
    sub_B96E90((__int64)&v39, v15, 1);
    v13 = v37;
  }
  v40 = *(_DWORD *)(a2 + 72);
  if ( !v13 )
  {
    v47 = sub_3007260((__int64)&v37);
    v48 = v16;
    v17 = v47;
    v18 = v48;
    goto LABEL_18;
  }
  if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
LABEL_45:
    BUG();
  v30 = 16LL * (v13 - 1);
  v17 = *(_QWORD *)&byte_444C4A0[v30];
  v18 = byte_444C4A0[v30 + 8];
LABEL_18:
  v43 = v17;
  LOBYTE(v44) = v18;
  v19 = sub_CA1930(&v43);
  v42 = v19;
  if ( v19 > 0x40 )
  {
    sub_C43690((__int64)&v41, 1, 0);
    v20 = a1[1];
    v44 = v42;
    if ( v42 > 0x40 )
    {
      v32 = v20;
      sub_C43780((__int64)&v43, (const void **)&v41);
      v20 = v32;
      goto LABEL_21;
    }
  }
  else
  {
    v41 = 1;
    v20 = a1[1];
    v44 = v19;
  }
  v43 = v41;
LABEL_21:
  sub_3401900(v20, (__int64)&v39, v37, v38, (__int64)&v43, 1, a5);
  v23 = v21;
  if ( v44 > 0x40 && v43 )
  {
    v33 = v21;
    j_j___libc_free_0_0(v43);
    v23 = v33;
  }
  v34 = v23;
  v24 = sub_33FAF80(a1[1], 214, (__int64)&v39, v8, v7, v22, a5);
  *((_QWORD *)&v31 + 1) = v25 | v34 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v31 = v24;
  v27 = sub_3406EB0((_QWORD *)a1[1], 0x3Au, (__int64)&v39, v8, v7, v26, v31, *(_OWORD *)*(_QWORD *)(a2 + 40));
  sub_375BC20(a1, (__int64)v27, v28, a3, a4, a5);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
}
