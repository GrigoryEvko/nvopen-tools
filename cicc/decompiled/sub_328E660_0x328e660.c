// Function: sub_328E660
// Address: 0x328e660
//
__int64 __fastcall sub_328E660(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 *v6; // rax
  __int64 v7; // rbx
  int v8; // edx
  __int64 v9; // r14
  int v10; // edx
  __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int16 v16; // dx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rcx
  _QWORD *v23; // rbx
  __int16 v24; // ax
  __int64 v26; // rax
  bool v27; // zf
  __int16 v28; // [rsp+4h] [rbp-9Ch]
  int v29; // [rsp+10h] [rbp-90h]
  unsigned int v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+10h] [rbp-90h]
  __int64 v33; // [rsp+10h] [rbp-90h]
  int v35; // [rsp+2Ch] [rbp-74h] BYREF
  unsigned __int16 v36; // [rsp+30h] [rbp-70h] BYREF
  __int64 v37; // [rsp+38h] [rbp-68h]
  __int64 v38; // [rsp+40h] [rbp-60h] BYREF
  int v39; // [rsp+48h] [rbp-58h]
  unsigned __int64 v40; // [rsp+50h] [rbp-50h] BYREF
  __int64 v41; // [rsp+58h] [rbp-48h]
  __int64 v42; // [rsp+60h] [rbp-40h]
  __int64 v43; // [rsp+68h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 40);
  v7 = *v6;
  v8 = *(_DWORD *)(*v6 + 24);
  if ( v8 == 55 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL * *((unsigned int *)v6 + 2));
    v8 = *(_DWORD *)(v7 + 24);
  }
  v9 = v6[5];
  if ( v8 != 298 )
    v7 = 0;
  v10 = *(_DWORD *)(v9 + 24);
  if ( v10 == 55 )
  {
    v11 = *(_QWORD *)(v9 + 40) + 40LL * *((unsigned int *)v6 + 12);
    v9 = *(_QWORD *)v11;
    v10 = *(_DWORD *)(*(_QWORD *)v11 + 24LL);
  }
  v12 = *(__int64 **)(*a1 + 40LL);
  if ( v10 != 298 )
  {
    sub_2E79000(v12);
    return 0;
  }
  if ( *(_BYTE *)sub_2E79000(v12) )
  {
    if ( v7 )
    {
      v26 = v7;
      v7 = v9;
      v27 = *(_DWORD *)(v9 + 24) == 298;
      v9 = v26;
      if ( v27 )
        goto LABEL_11;
    }
    return 0;
  }
  if ( !v7 || *(_DWORD *)(v7 + 24) != 298 )
    return 0;
LABEL_11:
  if ( (*(_BYTE *)(v7 + 33) & 0xC) != 0 )
    return 0;
  if ( *(_DWORD *)(v9 + 24) != 298 )
    return 0;
  if ( (*(_BYTE *)(v9 + 33) & 0xC) != 0 )
    return 0;
  v13 = *(_QWORD *)(v7 + 56);
  if ( !v13 )
    return 0;
  if ( *(_QWORD *)(v13 + 32) )
    return 0;
  v14 = *(_QWORD *)(v9 + 56);
  if ( !v14 )
    return 0;
  if ( *(_QWORD *)(v14 + 32) )
    return 0;
  v29 = sub_2EAC1E0(*(_QWORD *)(v7 + 112));
  if ( (unsigned int)sub_2EAC1E0(*(_QWORD *)(v9 + 112)) != v29 )
    return 0;
  v15 = *(_QWORD *)(v7 + 48);
  v35 = 0;
  v16 = *(_WORD *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v36 = v16;
  v37 = v17;
  v40 = sub_3285A00(&v36);
  v41 = v18;
  v30 = sub_CA1930(&v40);
  if ( *((_BYTE *)a1 + 33) )
  {
    if ( !sub_328D6E0(a1[1], 0x12Au, a3) )
      return 0;
  }
  if ( !(unsigned __int8)sub_33D01F0(*a1, v9, v7, v30, 1) )
    return 0;
  v19 = a1[1];
  v31 = *(_QWORD *)(v7 + 112);
  v20 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
  if ( !(unsigned __int8)sub_2FEBB30(v19, *(_QWORD *)(*a1 + 64LL), v20, a3, a4, v31, &v35) || !v35 )
    return 0;
  v40 = 0;
  v21 = *a1;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  LOBYTE(v24) = sub_2EAC4F0(*(_QWORD *)(v7 + 112));
  v22 = *(_QWORD *)(v7 + 112);
  v23 = *(_QWORD **)(v7 + 40);
  v38 = *(_QWORD *)(a2 + 80);
  HIBYTE(v24) = 1;
  if ( v38 )
  {
    v28 = v24;
    v32 = v22;
    sub_325F5D0(&v38);
    v24 = v28;
    v22 = v32;
  }
  v39 = *(_DWORD *)(a2 + 72);
  v33 = sub_33F1F00(
          v21,
          a3,
          a4,
          (unsigned int)&v38,
          *v23,
          v23[1],
          v23[5],
          v23[6],
          *(_OWORD *)v22,
          *(_QWORD *)(v22 + 16),
          v24,
          0,
          (__int64)&v40,
          0);
  sub_9C6650(&v38);
  return v33;
}
