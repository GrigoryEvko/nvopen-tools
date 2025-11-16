// Function: sub_33829D0
// Address: 0x33829d0
//
void __fastcall sub_33829D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 *v7; // rdi
  __int64 v8; // r12
  int v9; // eax
  __int64 *v10; // r12
  __int64 *v11; // r13
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rdx
  int v23; // ecx
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rsi
  __int64 v27; // r12
  int v28; // edx
  int v29; // r14d
  _QWORD *v30; // rax
  __int128 v31; // [rsp-10h] [rbp-F0h]
  __int64 v33; // [rsp+20h] [rbp-C0h]
  __int64 v34; // [rsp+20h] [rbp-C0h]
  __int64 v35; // [rsp+28h] [rbp-B8h]
  __int64 v36; // [rsp+28h] [rbp-B8h]
  __int64 v37; // [rsp+48h] [rbp-98h] BYREF
  __int64 v38; // [rsp+50h] [rbp-90h] BYREF
  int v39; // [rsp+58h] [rbp-88h]
  __int64 *v40; // [rsp+60h] [rbp-80h] BYREF
  __int64 v41; // [rsp+68h] [rbp-78h]
  _BYTE v42[16]; // [rsp+70h] [rbp-70h] BYREF
  _BYTE *v43; // [rsp+80h] [rbp-60h] BYREF
  __int64 v44; // [rsp+88h] [rbp-58h]
  _BYTE v45[80]; // [rsp+90h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a1[108] + 64);
  sub_B15700((__int64)&v43, a2, a3, 0);
  sub_B6EB20(v4, (__int64)&v43);
  v5 = a1[108];
  v6 = *(_QWORD *)(a2 + 8);
  v40 = (__int64 *)v42;
  v7 = *(__int64 **)(v5 + 40);
  v8 = *(_QWORD *)(v5 + 16);
  v41 = 0x100000000LL;
  v9 = sub_2E79000(v7);
  LOBYTE(v44) = 0;
  *((_QWORD *)&v31 + 1) = v44;
  v43 = 0;
  *(_QWORD *)&v31 = 0;
  sub_34B8C80(v8, v9, v6, (unsigned int)&v40, 0, 0, v31);
  if ( (_DWORD)v41 )
  {
    v44 = 0x100000000LL;
    v43 = v45;
    v10 = &v40[2 * (unsigned int)v41];
    v11 = v40;
    do
    {
      v12 = v11[1];
      v13 = *v11;
      v14 = a1[108];
      v38 = 0;
      v39 = 0;
      v15 = sub_33F17F0(v14, 51, &v38, v13, v12);
      v17 = v15;
      v18 = v16;
      if ( v38 )
      {
        v33 = v15;
        v35 = v16;
        sub_B91220((__int64)&v38, v38);
        v17 = v33;
        v18 = v35;
      }
      v19 = (unsigned int)v44;
      v20 = (unsigned int)v44 + 1LL;
      if ( v20 > HIDWORD(v44) )
      {
        v34 = v17;
        v36 = v18;
        sub_C8D5F0((__int64)&v43, v45, v20, 0x10u, v17, v18);
        v19 = (unsigned int)v44;
        v17 = v34;
        v18 = v36;
      }
      v21 = (__int64 *)&v43[16 * v19];
      v11 += 2;
      *v21 = v17;
      v21[1] = v18;
      v22 = (unsigned int)(v44 + 1);
      LODWORD(v44) = v44 + 1;
    }
    while ( v10 != v11 );
    v23 = *((_DWORD *)a1 + 212);
    v24 = *a1;
    v38 = 0;
    v25 = a1[108];
    v39 = v23;
    if ( v24 )
    {
      if ( &v38 != (__int64 *)(v24 + 48) )
      {
        v26 = *(_QWORD *)(v24 + 48);
        v38 = v26;
        if ( v26 )
        {
          sub_B96E90((__int64)&v38, v26, 1);
          v22 = (unsigned int)v44;
        }
      }
    }
    v27 = sub_3411660(v25, v43, v22, &v38);
    v29 = v28;
    v37 = a2;
    v30 = sub_337DC20((__int64)(a1 + 1), &v37);
    *v30 = v27;
    *((_DWORD *)v30 + 2) = v29;
    if ( v38 )
      sub_B91220((__int64)&v38, v38);
    if ( v43 != v45 )
      _libc_free((unsigned __int64)v43);
  }
  if ( v40 != (__int64 *)v42 )
    _libc_free((unsigned __int64)v40);
}
