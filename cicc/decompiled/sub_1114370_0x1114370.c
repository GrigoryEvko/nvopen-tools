// Function: sub_1114370
// Address: 0x1114370
//
_QWORD *__fastcall sub_1114370(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned int v6; // edx
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rbx
  _QWORD *v13; // r12
  _QWORD **v14; // rdx
  int v15; // ecx
  int v16; // eax
  __int64 *v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // rdi
  unsigned int v20; // r15d
  __int64 v21; // rbx
  unsigned __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // r14
  _QWORD *v26; // rax
  __int64 v27; // rdi
  unsigned __int64 v28; // rax
  unsigned int v29; // eax
  __int64 v30; // rdi
  __int64 v31; // r14
  _QWORD **v32; // rdx
  int v33; // ecx
  int v34; // eax
  __int64 *v35; // rax
  __int64 v36; // rsi
  unsigned int v38; // edx
  unsigned int v39; // eax
  __int64 v40; // rdi
  __int64 v41; // r14
  _QWORD *v42; // rax
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v49; // [rsp+28h] [rbp-78h]
  __int64 v50; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v51; // [rsp+38h] [rbp-68h]
  __int64 v52; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v53; // [rsp+48h] [rbp-58h]
  __int16 v54; // [rsp+60h] [rbp-40h]

  v6 = *(_DWORD *)(a3 + 8);
  if ( (unsigned int)(a4 - 36) <= 1 )
  {
    v51 = v6;
    if ( v6 > 0x40 )
    {
      sub_C43690((__int64)&v50, -1, 1);
    }
    else
    {
      v28 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
      if ( !v6 )
        v28 = 0;
      v50 = v28;
    }
    sub_C46B40((__int64)&v50, (__int64 *)a3);
    v29 = v51;
    v30 = *(_QWORD *)(a2 + 8);
    v51 = 0;
    v53 = v29;
    v52 = v50;
    v31 = sub_AD8D80(v30, (__int64)&v52);
    if ( v53 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    v54 = 257;
    v13 = sub_BD2C40(72, unk_3F10FD0);
    if ( v13 )
    {
      v32 = *(_QWORD ***)(a2 + 8);
      v33 = *((unsigned __int8 *)v32 + 8);
      if ( (unsigned int)(v33 - 17) > 1 )
      {
        v36 = sub_BCB2A0(*v32);
      }
      else
      {
        v34 = *((_DWORD *)v32 + 8);
        BYTE4(v50) = (_BYTE)v33 == 18;
        LODWORD(v50) = v34;
        v35 = (__int64 *)sub_BCB2A0(*v32);
        v36 = sub_BCE1B0(v35, v50);
      }
      sub_B523C0((__int64)v13, v36, 53, 34, a2, v31, (__int64)&v52, 0, 0, 0);
    }
    return v13;
  }
  if ( (unsigned int)(a4 - 34) <= 1 )
  {
    v49 = v6;
    if ( v6 > 0x40 )
    {
      sub_C43780((__int64)&v48, (const void **)a3);
      v6 = v49;
      if ( v49 > 0x40 )
      {
        sub_C43D10((__int64)&v48);
        goto LABEL_8;
      }
      v8 = v48;
    }
    else
    {
      v8 = *(_QWORD *)a3;
    }
    v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v8;
    if ( !v6 )
      v9 = 0;
    v48 = v9;
LABEL_8:
    sub_C46250((__int64)&v48);
    v10 = v49;
    v11 = *(_QWORD *)(a2 + 8);
    v49 = 0;
    v51 = v10;
    v50 = v48;
    v12 = sub_AD8D80(v11, (__int64)&v50);
    v54 = 257;
    v13 = sub_BD2C40(72, unk_3F10FD0);
    if ( v13 )
    {
      v14 = *(_QWORD ***)(a2 + 8);
      v15 = *((unsigned __int8 *)v14 + 8);
      if ( (unsigned int)(v15 - 17) > 1 )
      {
        v18 = sub_BCB2A0(*v14);
      }
      else
      {
        v16 = *((_DWORD *)v14 + 8);
        BYTE4(v46) = (_BYTE)v15 == 18;
        LODWORD(v46) = v16;
        v17 = (__int64 *)sub_BCB2A0(*v14);
        v18 = sub_BCE1B0(v17, v46);
      }
      sub_B523C0((__int64)v13, v18, 53, 36, a2, v12, (__int64)&v52, 0, 0, 0);
    }
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( v49 > 0x40 )
    {
      v19 = v48;
      if ( v48 )
        goto LABEL_17;
    }
    return v13;
  }
  v20 = v6 - 1;
  v45 = v6;
  v21 = ~(1LL << ((unsigned __int8)v6 - 1));
  if ( v6 <= 0x40 )
  {
    v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    if ( !v6 )
      v22 = 0;
    v44 = v22;
    goto LABEL_22;
  }
  sub_C43690((__int64)&v44, -1, 1);
  if ( v45 <= 0x40 )
  {
LABEL_22:
    v44 &= v21;
    goto LABEL_23;
  }
  *(_QWORD *)(v44 + 8LL * (v20 >> 6)) &= v21;
LABEL_23:
  if ( (unsigned int)(a4 - 40) > 1 )
  {
    v47 = *(_DWORD *)(a3 + 8);
    if ( v47 > 0x40 )
      sub_C43780((__int64)&v46, (const void **)a3);
    else
      v46 = *(_QWORD *)a3;
    sub_C46F20((__int64)&v46, 1u);
    v38 = v47;
    v47 = 0;
    v49 = v38;
    v48 = v46;
    if ( v38 <= 0x40 )
    {
      v43 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v38) & ~v46;
      if ( !v38 )
        v43 = 0;
      v48 = v43;
    }
    else
    {
      sub_C43D10((__int64)&v48);
    }
    sub_C46250((__int64)&v48);
    sub_C45EE0((__int64)&v48, (__int64 *)&v44);
    v39 = v49;
    v40 = *(_QWORD *)(a2 + 8);
    v49 = 0;
    v51 = v39;
    v50 = v48;
    v41 = sub_AD8D80(v40, (__int64)&v50);
    v54 = 257;
    v42 = sub_BD2C40(72, unk_3F10FD0);
    v13 = v42;
    if ( v42 )
      sub_1113300((__int64)v42, 40, a2, v41, (__int64)&v52);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
    if ( v47 > 0x40 )
    {
      v27 = v46;
      if ( v46 )
        goto LABEL_33;
    }
  }
  else
  {
    v49 = v45;
    if ( v45 > 0x40 )
      sub_C43780((__int64)&v48, (const void **)&v44);
    else
      v48 = v44;
    sub_C46B40((__int64)&v48, (__int64 *)a3);
    v23 = v49;
    v24 = *(_QWORD *)(a2 + 8);
    v49 = 0;
    v51 = v23;
    v50 = v48;
    v25 = sub_AD8D80(v24, (__int64)&v50);
    v54 = 257;
    v26 = sub_BD2C40(72, unk_3F10FD0);
    v13 = v26;
    if ( v26 )
      sub_1113300((__int64)v26, 38, a2, v25, (__int64)&v52);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( v49 > 0x40 )
    {
      v27 = v48;
      if ( v48 )
LABEL_33:
        j_j___libc_free_0_0(v27);
    }
  }
  if ( v45 > 0x40 )
  {
    v19 = v44;
    if ( v44 )
LABEL_17:
      j_j___libc_free_0_0(v19);
  }
  return v13;
}
