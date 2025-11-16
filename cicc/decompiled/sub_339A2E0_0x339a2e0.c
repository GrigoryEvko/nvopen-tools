// Function: sub_339A2E0
// Address: 0x339a2e0
//
void __fastcall sub_339A2E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r14
  int v5; // eax
  __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // r12d
  _BYTE *v13; // rax
  _BYTE *v14; // rdx
  _BYTE *i; // rdx
  int v16; // edx
  __int64 v17; // r9
  int v18; // eax
  __int64 v19; // r13
  unsigned int v20; // r12d
  int v21; // ebx
  __int64 v22; // r15
  __int64 v23; // rax
  _BYTE *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int16 *v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rcx
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // r12
  __int64 v34; // r13
  int v35; // eax
  int v36; // edx
  int v37; // r9d
  int v38; // ecx
  int v39; // r8d
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // r12
  int v43; // edx
  int v44; // r13d
  _QWORD *v45; // rax
  _BYTE *v46; // rdi
  __int64 v47; // rdi
  __int64 v48; // r12
  int v49; // edx
  int v50; // r13d
  _QWORD *v51; // rax
  __int128 v52; // [rsp-10h] [rbp-160h]
  __int128 v53; // [rsp-10h] [rbp-160h]
  int v54; // [rsp+10h] [rbp-140h]
  unsigned __int8 v55; // [rsp+18h] [rbp-138h]
  __int64 v56; // [rsp+18h] [rbp-138h]
  int v58; // [rsp+30h] [rbp-120h]
  __int64 v59; // [rsp+30h] [rbp-120h]
  int v60; // [rsp+30h] [rbp-120h]
  __int64 v61; // [rsp+38h] [rbp-118h]
  int v62; // [rsp+38h] [rbp-118h]
  __int64 v63; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v64; // [rsp+70h] [rbp-E0h] BYREF
  int v65; // [rsp+78h] [rbp-D8h]
  _BYTE *v66; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v67; // [rsp+88h] [rbp-C8h]
  _BYTE v68[64]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v69; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v70; // [rsp+D8h] [rbp-78h]
  _BYTE v71[112]; // [rsp+E0h] [rbp-70h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a2 - 32);
  v4 = *(_QWORD *)(a2 + 8);
  v55 = *(_BYTE *)v3;
  v5 = sub_34B8B90(
         *(_QWORD *)(v3 + 8),
         *(_QWORD *)(a2 + 72),
         *(_QWORD *)(a2 + 72) + 4LL * *(unsigned int *)(a2 + 80),
         0);
  v67 = 0x400000000LL;
  v58 = v5;
  v6 = *(_QWORD *)(a1 + 864);
  v7 = *(__int64 **)(v6 + 40);
  v8 = *(_QWORD *)(v6 + 16);
  v66 = v68;
  v9 = sub_2E79000(v7);
  LOBYTE(v70) = 0;
  *((_QWORD *)&v52 + 1) = v70;
  v69 = 0;
  *(_QWORD *)&v52 = 0;
  sub_34B8C80(v8, v9, v4, (unsigned int)&v66, 0, 0, v52);
  v12 = v67;
  if ( (_DWORD)v67 )
  {
    v70 = 0x400000000LL;
    v61 = (unsigned int)v67;
    v13 = v71;
    v14 = v71;
    v69 = v71;
    if ( (unsigned int)v67 > 4 )
    {
      sub_C8D5F0((__int64)&v69, v71, (unsigned int)v67, 0x10u, v10, v11);
      v14 = v69;
      v13 = &v69[16 * (unsigned int)v70];
    }
    for ( i = &v14[16 * v12]; i != v13; v13 += 16 )
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = 0;
        *((_DWORD *)v13 + 2) = 0;
      }
    }
    LODWORD(v70) = v12;
    v17 = sub_338B750(v2, v3);
    if ( v12 )
    {
      v18 = v58;
      v19 = 0;
      v59 = v2;
      v20 = v55 - 12;
      v21 = v18 + v16;
      v22 = v17;
      do
      {
        v25 = (unsigned int)(v21 + v19);
        v26 = v22;
        if ( v20 <= 1 )
        {
          v27 = (unsigned __int16 *)(*(_QWORD *)(v22 + 48) + 16 * v25);
          v28 = *((_QWORD *)v27 + 1);
          v29 = *v27;
          v30 = *(_QWORD *)(v59 + 864);
          v64 = 0;
          v65 = 0;
          v31 = sub_33F17F0(v30, 51, &v64, v29, v28);
          v26 = v31;
          if ( v64 )
          {
            v54 = v25;
            v56 = v31;
            sub_B91220((__int64)&v64, v64);
            LODWORD(v25) = v54;
            v26 = v56;
          }
        }
        v23 = v19++;
        v24 = &v69[16 * v23];
        *(_QWORD *)v24 = v26;
        *((_DWORD *)v24 + 2) = v25;
      }
      while ( v61 != v19 );
      v2 = v59;
    }
    v32 = *(_QWORD *)(v2 + 864);
    v33 = (__int64)v69;
    v34 = (unsigned int)v70;
    v35 = sub_33E5830(v32, v66);
    v64 = 0;
    v38 = v35;
    v39 = v36;
    v40 = *(_QWORD *)v2;
    v65 = *(_DWORD *)(v2 + 848);
    if ( v40 )
    {
      if ( &v64 != (__int64 *)(v40 + 48) )
      {
        v41 = *(_QWORD *)(v40 + 48);
        v64 = v41;
        if ( v41 )
        {
          v60 = v36;
          v62 = v38;
          sub_B96E90((__int64)&v64, v41, 1);
          v39 = v60;
          v38 = v62;
        }
      }
    }
    *((_QWORD *)&v53 + 1) = v34;
    *(_QWORD *)&v53 = v33;
    v42 = sub_3411630(v32, 55, (unsigned int)&v64, v38, v39, v37, v53);
    v44 = v43;
    v63 = a2;
    v45 = sub_337DC20(v2 + 8, &v63);
    *v45 = v42;
    *((_DWORD *)v45 + 2) = v44;
    if ( v64 )
      sub_B91220((__int64)&v64, v64);
    if ( v69 != v71 )
      _libc_free((unsigned __int64)v69);
    v46 = v66;
    if ( v66 != v68 )
LABEL_24:
      _libc_free((unsigned __int64)v46);
  }
  else
  {
    v47 = *(_QWORD *)(v2 + 864);
    v69 = 0;
    LODWORD(v70) = 0;
    v48 = sub_33F17F0(v47, 51, &v69, 1, 0);
    v50 = v49;
    if ( v69 )
      sub_B91220((__int64)&v69, (__int64)v69);
    v69 = (_BYTE *)a2;
    v51 = sub_337DC20(v2 + 8, (__int64 *)&v69);
    *v51 = v48;
    v46 = v66;
    *((_DWORD *)v51 + 2) = v50;
    if ( v46 != v68 )
      goto LABEL_24;
  }
}
