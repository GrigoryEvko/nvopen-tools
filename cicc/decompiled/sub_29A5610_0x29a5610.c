// Function: sub_29A5610
// Address: 0x29a5610
//
_QWORD *__fastcall sub_29A5610(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r15
  unsigned int *v9; // rax
  int v10; // ecx
  unsigned int *v11; // rdx
  __int64 v12; // rdx
  __int64 **v13; // rax
  __int64 **v14; // r15
  __int64 (__fastcall *v15)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // r12
  int v20; // r15d
  __int64 v21; // r15
  unsigned int *v22; // r15
  unsigned int *v23; // rbx
  __int64 v24; // rdx
  unsigned int v25; // esi
  unsigned __int64 v26; // rsi
  _BYTE v28[32]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v29; // [rsp+50h] [rbp-100h]
  _QWORD v30[4]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v31; // [rsp+80h] [rbp-D0h]
  unsigned int *v32; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+98h] [rbp-B8h]
  _BYTE v34[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+C0h] [rbp-90h]
  __int64 v36; // [rsp+C8h] [rbp-88h]
  __int64 v37; // [rsp+D0h] [rbp-80h]
  __int64 v38; // [rsp+D8h] [rbp-78h]
  void **v39; // [rsp+E0h] [rbp-70h]
  void **v40; // [rsp+E8h] [rbp-68h]
  __int64 v41; // [rsp+F0h] [rbp-60h]
  int v42; // [rsp+F8h] [rbp-58h]
  __int16 v43; // [rsp+FCh] [rbp-54h]
  char v44; // [rsp+FEh] [rbp-52h]
  __int64 v45; // [rsp+100h] [rbp-50h]
  __int64 v46; // [rsp+108h] [rbp-48h]
  void *v47; // [rsp+110h] [rbp-40h] BYREF
  void *v48; // [rsp+118h] [rbp-38h] BYREF

  v44 = 7;
  v38 = sub_BD5C60(a1);
  v39 = &v47;
  v40 = &v48;
  v33 = 0x200000000LL;
  v47 = &unk_49DA100;
  v32 = (unsigned int *)v34;
  v41 = 0;
  v48 = &unk_49DA0B0;
  v4 = *(_QWORD *)(a1 + 40);
  v42 = 0;
  v35 = v4;
  v43 = 512;
  v45 = 0;
  v46 = 0;
  v36 = a1 + 24;
  LOWORD(v37) = 0;
  v5 = *(_QWORD *)sub_B46C60(a1);
  v30[0] = v5;
  if ( !v5 || (sub_B96E90((__int64)v30, v5, 1), (v8 = v30[0]) == 0) )
  {
    sub_93FB40((__int64)&v32, 0);
    v8 = v30[0];
    goto LABEL_21;
  }
  v9 = v32;
  v10 = v33;
  v11 = &v32[4 * (unsigned int)v33];
  if ( v32 == v11 )
  {
LABEL_23:
    if ( (unsigned int)v33 >= (unsigned __int64)HIDWORD(v33) )
    {
      v26 = (unsigned int)v33 + 1LL;
      if ( HIDWORD(v33) < v26 )
      {
        sub_C8D5F0((__int64)&v32, v34, v26, 0x10u, v6, v7);
        v11 = &v32[4 * (unsigned int)v33];
      }
      *(_QWORD *)v11 = 0;
      *((_QWORD *)v11 + 1) = v8;
      v8 = v30[0];
      LODWORD(v33) = v33 + 1;
    }
    else
    {
      if ( v11 )
      {
        *v11 = 0;
        *((_QWORD *)v11 + 1) = v8;
        v10 = v33;
        v8 = v30[0];
      }
      LODWORD(v33) = v10 + 1;
    }
LABEL_21:
    if ( !v8 )
      goto LABEL_9;
    goto LABEL_8;
  }
  while ( *v9 )
  {
    v9 += 4;
    if ( v11 == v9 )
      goto LABEL_23;
  }
  *((_QWORD *)v9 + 1) = v30[0];
LABEL_8:
  sub_B91220((__int64)v30, v8);
LABEL_9:
  v12 = *(_QWORD *)(a1 - 32);
  v13 = *(__int64 ***)(a2 + 8);
  if ( *(__int64 ***)(v12 + 8) == v13 )
    goto LABEL_17;
  v29 = 257;
  v14 = *(__int64 ***)(v12 + 8);
  if ( v14 == v13 )
    goto LABEL_17;
  v15 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v39 + 15);
  if ( v15 != sub_920130 )
  {
    v16 = v15((__int64)v39, 49u, (_BYTE *)a2, *(_QWORD *)(v12 + 8));
    goto LABEL_15;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x31u) )
      v16 = sub_ADAB70(49, a2, v14, 0);
    else
      v16 = sub_AA93C0(0x31u, a2, (__int64)v14);
LABEL_15:
    if ( v16 )
    {
      v12 = *(_QWORD *)(a1 - 32);
      a2 = v16;
      goto LABEL_17;
    }
  }
  v31 = 257;
  a2 = sub_B51D30(49, a2, (__int64)v14, (__int64)v30, 0, 0);
  if ( (unsigned __int8)sub_920620(a2) )
  {
    v20 = v42;
    if ( v41 )
      sub_B99FD0(a2, 3u, v41);
    sub_B45150(a2, v20);
  }
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v40 + 2))(v40, a2, v28, v36, v37);
  v21 = 4LL * (unsigned int)v33;
  if ( v32 != &v32[v21] )
  {
    v22 = &v32[v21];
    v23 = v32;
    do
    {
      v24 = *((_QWORD *)v23 + 1);
      v25 = *v23;
      v23 += 4;
      sub_B99FD0(a2, v25, v24);
    }
    while ( v22 != v23 );
  }
  v12 = *(_QWORD *)(a1 - 32);
LABEL_17:
  v31 = 257;
  v17 = sub_92B530(&v32, 0x20u, v12, (_BYTE *)a2, (__int64)v30);
  v18 = sub_29A49A0(a1, v17, a3);
  nullsub_61();
  v47 = &unk_49DA100;
  nullsub_63();
  if ( v32 != (unsigned int *)v34 )
    _libc_free((unsigned __int64)v32);
  return v18;
}
