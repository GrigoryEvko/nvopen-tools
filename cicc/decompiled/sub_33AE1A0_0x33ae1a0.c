// Function: sub_33AE1A0
// Address: 0x33ae1a0
//
void __fastcall sub_33AE1A0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 *v4; // rdi
  int v5; // eax
  __int64 v6; // rdi
  int v7; // r12d
  unsigned __int64 v8; // rdi
  _BYTE *v9; // rax
  _BYTE *v10; // rcx
  _BYTE *i; // rdx
  int v12; // edx
  int v13; // eax
  int v14; // r13d
  __int64 v15; // r12
  __int64 v16; // rax
  int v17; // r9d
  _BYTE *v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // edx
  int v23; // edi
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // r14
  unsigned __int64 v27; // r12
  __int64 v28; // r13
  int v29; // eax
  int v30; // edx
  int v31; // r9d
  int v32; // ecx
  int v33; // r8d
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // r12
  int v37; // edx
  int v38; // r13d
  _QWORD *v39; // rax
  __int64 v40; // [rsp-10h] [rbp-170h]
  __int128 v41; // [rsp-10h] [rbp-170h]
  __int64 v42; // [rsp-8h] [rbp-168h]
  _BYTE *v44; // [rsp+30h] [rbp-130h]
  int v45; // [rsp+38h] [rbp-128h]
  int v46; // [rsp+40h] [rbp-120h]
  int v47; // [rsp+40h] [rbp-120h]
  int v48; // [rsp+48h] [rbp-118h]
  int v49; // [rsp+48h] [rbp-118h]
  __int64 v50; // [rsp+78h] [rbp-E8h] BYREF
  __int64 v51; // [rsp+80h] [rbp-E0h] BYREF
  int v52; // [rsp+88h] [rbp-D8h]
  _BYTE *v53; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v54; // [rsp+98h] [rbp-C8h]
  _BYTE v55[64]; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE *v56; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v57; // [rsp+E8h] [rbp-78h]
  _BYTE v58[112]; // [rsp+F0h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL);
  v53 = v55;
  v54 = 0x400000000LL;
  v5 = sub_2E79000(v4);
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v56 = 0;
  LOBYTE(v57) = 0;
  sub_34B8C80(v6, v5, v3, (unsigned int)&v53, 0, 0, __PAIR128__(v57, 0));
  v7 = v54;
  if ( !(_DWORD)v54 )
  {
    v8 = (unsigned __int64)v53;
    if ( v53 == v55 )
      return;
    goto LABEL_3;
  }
  v9 = v58;
  v57 = 0x400000000LL;
  v10 = v58;
  v56 = v58;
  if ( (unsigned int)v54 > 4 )
  {
    sub_C8D5F0((__int64)&v56, v58, (unsigned int)v54, 0x10u, v40, v42);
    v10 = v56;
    v9 = &v56[16 * (unsigned int)v57];
  }
  for ( i = &v10[16 * v7]; i != v9; v9 += 16 )
  {
    if ( v9 )
    {
      *(_QWORD *)v9 = 0;
      *((_DWORD *)v9 + 2) = 0;
    }
  }
  LODWORD(v57) = v7;
  v45 = sub_338B750(a1, *(_QWORD *)(a2 - 32));
  v13 = v7 + v12;
  v14 = v12;
  v15 = 0;
  v46 = v13;
  do
  {
    v16 = *(_QWORD *)(a1 + 864);
    v51 = 0;
    v17 = v45;
    v48 = v16;
    v18 = &v53[v15];
    v19 = *(_QWORD *)a1;
    v52 = *(_DWORD *)(a1 + 848);
    if ( v19 )
    {
      if ( &v51 != (__int64 *)(v19 + 48) )
      {
        v20 = *(_QWORD *)(v19 + 48);
        v51 = v20;
        if ( v20 )
        {
          v44 = &v53[v15];
          sub_B96E90((__int64)&v51, v20, 1);
          v17 = v45;
          v18 = v44;
        }
      }
    }
    v21 = sub_33FAF80(v48, 52, (unsigned int)&v51, *(_DWORD *)v18, *((_QWORD *)v18 + 1), v17);
    v23 = v22;
    v24 = v21;
    v25 = (unsigned __int64)v56;
    *(_QWORD *)&v56[v15] = v24;
    *(_DWORD *)(v25 + v15 + 8) = v23;
    if ( v51 )
      sub_B91220((__int64)&v51, v51);
    ++v14;
    v15 += 16;
  }
  while ( v14 != v46 );
  v26 = *(_QWORD *)(a1 + 864);
  v27 = (unsigned __int64)v56;
  v28 = (unsigned int)v57;
  v29 = sub_33E5830(v26, v53);
  v51 = 0;
  v32 = v29;
  v33 = v30;
  v34 = *(_QWORD *)a1;
  v52 = *(_DWORD *)(a1 + 848);
  if ( v34 )
  {
    if ( &v51 != (__int64 *)(v34 + 48) )
    {
      v35 = *(_QWORD *)(v34 + 48);
      v51 = v35;
      if ( v35 )
      {
        v47 = v30;
        v49 = v32;
        sub_B96E90((__int64)&v51, v35, 1);
        v33 = v47;
        v32 = v49;
      }
    }
  }
  *((_QWORD *)&v41 + 1) = v28;
  *(_QWORD *)&v41 = v27;
  v36 = sub_3411630(v26, 55, (unsigned int)&v51, v32, v33, v31, v41);
  v38 = v37;
  v50 = a2;
  v39 = sub_337DC20(a1 + 8, &v50);
  *v39 = v36;
  *((_DWORD *)v39 + 2) = v38;
  if ( v51 )
    sub_B91220((__int64)&v51, v51);
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  v8 = (unsigned __int64)v53;
  if ( v53 != v55 )
LABEL_3:
    _libc_free(v8);
}
