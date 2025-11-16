// Function: sub_17767C0
// Address: 0x17767c0
//
void __fastcall sub_17767C0(__int64 a1, __int64 a2, __int64 **a3)
{
  __int64 ***v4; // r13
  __int64 **v5; // rax
  unsigned int v6; // r15d
  int v7; // r15d
  bool v8; // zf
  unsigned int v9; // ecx
  __int64 v10; // r12
  __int64 **v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r14
  __int64 v16; // rdi
  __int64 *v17; // r13
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 *v20; // rsi
  __int64 ***v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  char v26; // cl
  int v27; // eax
  _BYTE *v28; // rbx
  __int16 v29; // dx
  __int16 v30; // si
  __int64 v31; // r12
  _BYTE *v32; // r12
  int v33; // ecx
  __int64 v34; // rax
  unsigned __int64 *v35; // r15
  __int64 **v36; // rax
  unsigned __int64 v37; // rcx
  __int64 v38; // rsi
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  unsigned int v41; // [rsp+10h] [rbp-140h]
  unsigned __int8 v42; // [rsp+14h] [rbp-13Ch]
  __int64 ***v44; // [rsp+20h] [rbp-130h] BYREF
  __int64 v45; // [rsp+28h] [rbp-128h] BYREF
  __int64 v46[2]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v47; // [rsp+40h] [rbp-110h]
  __int64 v48[2]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v49; // [rsp+60h] [rbp-F0h]
  _QWORD v50[2]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v51; // [rsp+80h] [rbp-D0h]
  _BYTE *v52; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+98h] [rbp-B8h]
  _BYTE v54[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v4 = *(__int64 ****)(a2 - 24);
  v5 = *v4;
  if ( *((_BYTE *)*v4 + 8) == 16 )
    v5 = (__int64 **)*v5[2];
  v6 = *((_DWORD *)v5 + 2);
  v52 = v54;
  v7 = v6 >> 8;
  v8 = *(_QWORD *)(a2 + 48) == 0;
  v53 = 0x800000000LL;
  if ( !v8 || (v9 = *(unsigned __int16 *)(a2 + 18), (v9 & 0x8000u) != 0) )
  {
    sub_161F840(a2, (__int64)&v52);
    v9 = *(unsigned __int16 *)(a2 + 18);
  }
  v10 = *(_QWORD *)(a1 + 8);
  v47 = 257;
  v42 = v9 & 1;
  v41 = 1 << (v9 >> 1) >> 1;
  v11 = (__int64 **)sub_1647190(*a3, v7);
  if ( v11 != *v4 )
  {
    if ( *((_BYTE *)v4 + 16) > 0x10u )
    {
      v51 = 257;
      v4 = (__int64 ***)sub_15FDBD0(47, (__int64)v4, (__int64)v11, (__int64)v50, 0);
      v34 = *(_QWORD *)(v10 + 8);
      if ( v34 )
      {
        v35 = *(unsigned __int64 **)(v10 + 16);
        sub_157E9D0(v34 + 40, (__int64)v4);
        v36 = v4[3];
        v37 = *v35;
        v4[4] = (__int64 **)v35;
        v37 &= 0xFFFFFFFFFFFFFFF8LL;
        v4[3] = (__int64 **)(v37 | (unsigned __int8)v36 & 7);
        *(_QWORD *)(v37 + 8) = v4 + 3;
        *v35 = *v35 & 7 | (unsigned __int64)(v4 + 3);
      }
      v20 = v46;
      v21 = v4;
      sub_164B780((__int64)v4, v46);
      v8 = *(_QWORD *)(v10 + 80) == 0;
      v44 = v4;
      if ( v8 )
LABEL_39:
        sub_4263D6(v21, v20, v22);
      (*(void (__fastcall **)(__int64, __int64 ****))(v10 + 88))(v10 + 64, &v44);
      v38 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 )
      {
        v48[0] = *(_QWORD *)v10;
        sub_1623A60((__int64)v48, v38, 2);
        v39 = (__int64)v4[6];
        if ( v39 )
          sub_161E7C0((__int64)(v4 + 6), v39);
        v40 = (unsigned __int8 *)v48[0];
        v4[6] = (__int64 **)v48[0];
        if ( v40 )
          sub_1623210((__int64)v48, v40, (__int64)(v4 + 6));
      }
    }
    else
    {
      v12 = sub_15A46C0(47, v4, v11, 0);
      v13 = sub_14DBA30(v12, *(_QWORD *)(v10 + 96), 0);
      if ( v13 )
        v12 = v13;
      v4 = (__int64 ***)v12;
    }
  }
  v49 = 257;
  v14 = sub_1648A60(64, 2u);
  v15 = (__int64)v14;
  if ( v14 )
    sub_15F9650((__int64)v14, (__int64)a3, (__int64)v4, v42, 0);
  v16 = *(_QWORD *)(v10 + 8);
  if ( v16 )
  {
    v17 = *(__int64 **)(v10 + 16);
    sub_157E9D0(v16 + 40, v15);
    v18 = *(_QWORD *)(v15 + 24);
    v19 = *v17;
    *(_QWORD *)(v15 + 32) = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v15 + 24) = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v15 + 24;
    *v17 = *v17 & 7 | (v15 + 24);
  }
  v20 = v48;
  v21 = (__int64 ***)v15;
  sub_164B780(v15, v48);
  v8 = *(_QWORD *)(v10 + 80) == 0;
  v45 = v15;
  if ( v8 )
    goto LABEL_39;
  (*(void (__fastcall **)(__int64, __int64 *))(v10 + 88))(v10 + 64, &v45);
  v23 = *(_QWORD *)v10;
  if ( *(_QWORD *)v10 )
  {
    v50[0] = *(_QWORD *)v10;
    sub_1623A60((__int64)v50, v23, 2);
    v24 = *(_QWORD *)(v15 + 48);
    if ( v24 )
      sub_161E7C0(v15 + 48, v24);
    v25 = (unsigned __int8 *)v50[0];
    *(_QWORD *)(v15 + 48) = v50[0];
    if ( v25 )
      sub_1623210((__int64)v50, v25, v15 + 48);
  }
  sub_15F9450(v15, v41);
  v26 = *(_BYTE *)(a2 + 56);
  v27 = *(unsigned __int16 *)(a2 + 18) >> 7;
  v28 = v52;
  v29 = *(_WORD *)(v15 + 18) & 0x8000;
  v30 = *(_WORD *)(v15 + 18) & 0x7C7F;
  v31 = 16LL * (unsigned int)v53;
  *(_BYTE *)(v15 + 56) = v26;
  v32 = &v28[v31];
  *(_WORD *)(v15 + 18) = v29 | v30 | ((v27 & 7) << 7);
  if ( v28 != v32 )
  {
    do
    {
      v33 = *(_DWORD *)v28;
      if ( *(_DWORD *)v28 <= 0xAu && ((1LL << v33) & 0x7AF) != 0 )
        sub_1625C10(v15, v33, *((_QWORD *)v28 + 1));
      v28 += 16;
    }
    while ( v32 != v28 );
    v32 = v52;
  }
  if ( v32 != v54 )
    _libc_free((unsigned __int64)v32);
}
