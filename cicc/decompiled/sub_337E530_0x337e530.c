// Function: sub_337E530
// Address: 0x337e530
//
void __fastcall sub_337E530(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 *v6; // rdi
  int v7; // eax
  __int64 *v8; // r14
  __int64 v9; // r13
  unsigned int v10; // eax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // r10d
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  int v19; // edx
  int v20; // r14d
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  int v25; // edx
  int v26; // r14d
  _QWORD *v27; // rax
  _BYTE *v28; // rdi
  __int64 v29; // [rsp-10h] [rbp-140h]
  __int128 v30; // [rsp-10h] [rbp-140h]
  __int64 v31; // [rsp-8h] [rbp-138h]
  unsigned int v32; // [rsp+4h] [rbp-12Ch]
  __int64 v33; // [rsp+8h] [rbp-128h]
  __int64 v34; // [rsp+10h] [rbp-120h]
  int v35; // [rsp+18h] [rbp-118h]
  int v36; // [rsp+30h] [rbp-100h]
  unsigned int v37; // [rsp+38h] [rbp-F8h]
  __int64 v38; // [rsp+38h] [rbp-F8h]
  __int64 v39; // [rsp+50h] [rbp-E0h] BYREF
  int v40; // [rsp+58h] [rbp-D8h]
  __int64 v41; // [rsp+60h] [rbp-D0h] BYREF
  int v42; // [rsp+68h] [rbp-C8h]
  __int64 v43; // [rsp+70h] [rbp-C0h]
  __int64 v44; // [rsp+78h] [rbp-B8h]
  _QWORD v45[2]; // [rsp+80h] [rbp-B0h] BYREF
  _BYTE v46[32]; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 v47[2]; // [rsp+B0h] [rbp-80h] BYREF
  _BYTE v48[112]; // [rsp+C0h] [rbp-70h] BYREF

  v47[0] = (unsigned __int64)v48;
  v4 = *(_QWORD *)(a2 - 32);
  v5 = *(_QWORD *)(a2 + 8);
  v47[1] = 0x400000000LL;
  v45[1] = 0x400000000LL;
  v6 = *(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL);
  v45[0] = v46;
  v7 = sub_2E79000(v6);
  sub_34B8FD0(*(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL), v7, v5, (unsigned int)v47, 0, (unsigned int)v45, 0);
  v8 = (__int64 *)v47[0];
  v9 = *(_QWORD *)(a1 + 864);
  v10 = sub_35D5F90(*(_QWORD *)(a1 + 968), a2, *(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL), v4);
  v13 = *(unsigned int *)(a1 + 848);
  v14 = v29;
  v39 = 0;
  v15 = v10;
  v16 = *(_QWORD *)a1;
  v17 = v31;
  v40 = v13;
  if ( v16 )
  {
    v13 = v16 + 48;
    if ( &v39 != (__int64 *)(v16 + 48) )
    {
      v17 = *(_QWORD *)(v16 + 48);
      v39 = v17;
      if ( v17 )
      {
        v37 = v15;
        sub_B96E90((__int64)&v39, v17, 1);
        v15 = v37;
      }
    }
  }
  v32 = v15;
  v18 = sub_33738B0(a1, v17, v13, v14, v11, v12);
  v36 = v19;
  v33 = *v8;
  v34 = v8[1];
  v38 = v18;
  v20 = sub_33E5110(v9, (unsigned int)*v8, v34, 1, 0);
  v35 = v21;
  v41 = v38;
  v42 = v36;
  v22 = sub_33F0B60(v9, v32, (unsigned int)v33, v34);
  v44 = v23;
  *((_QWORD *)&v30 + 1) = 2;
  *(_QWORD *)&v30 = &v41;
  v43 = v22;
  v24 = sub_3411630(v9, 50, (unsigned int)&v39, v20, v35, (unsigned int)&v41, v30);
  v26 = v25;
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  v41 = a2;
  v27 = sub_337DC20(a1 + 8, &v41);
  *v27 = v24;
  v28 = (_BYTE *)v45[0];
  *((_DWORD *)v27 + 2) = v26;
  if ( v28 != v46 )
    _libc_free((unsigned __int64)v28);
  if ( (_BYTE *)v47[0] != v48 )
    _libc_free(v47[0]);
}
