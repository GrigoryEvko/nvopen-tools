// Function: sub_1B9EC50
// Address: 0x1b9ec50
//
void __fastcall sub_1B9EC50(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  char v5; // al
  __int64 *v6; // r15
  __int64 *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // r13d
  __int64 *v17; // r14
  __int64 *v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // r15
  __int64 *v21; // rax
  unsigned int v22; // ebx
  __int64 v23; // r15
  __int64 *v24; // rax
  unsigned int *v25; // r8
  int v26; // r9d
  __int64 v27; // rcx
  unsigned int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rdi
  unsigned __int64 v33; // rsi
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // r15
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 *v40; // [rsp+8h] [rbp-118h]
  __int64 *v41; // [rsp+8h] [rbp-118h]
  __int64 *v42; // [rsp+20h] [rbp-100h]
  __int64 v43; // [rsp+28h] [rbp-F8h]
  unsigned __int64 *v44; // [rsp+28h] [rbp-F8h]
  __int64 *v46; // [rsp+38h] [rbp-E8h]
  unsigned int *v47; // [rsp+38h] [rbp-E8h]
  __int64 v48; // [rsp+40h] [rbp-E0h]
  __int64 *v49; // [rsp+40h] [rbp-E0h]
  int v50; // [rsp+48h] [rbp-D8h]
  unsigned int v51; // [rsp+4Ch] [rbp-D4h]
  bool v52; // [rsp+59h] [rbp-C7h] BYREF
  bool v53; // [rsp+5Ah] [rbp-C6h] BYREF
  bool v54; // [rsp+5Bh] [rbp-C5h] BYREF
  int v55; // [rsp+5Ch] [rbp-C4h] BYREF
  __int64 **v56; // [rsp+60h] [rbp-C0h] BYREF
  _QWORD *v57; // [rsp+68h] [rbp-B8h] BYREF
  __int64 v58[2]; // [rsp+70h] [rbp-B0h] BYREF
  __int16 v59; // [rsp+80h] [rbp-A0h]
  unsigned __int64 v60[2]; // [rsp+90h] [rbp-90h] BYREF
  _BYTE v61[16]; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD v62[14]; // [rsp+B0h] [rbp-70h] BYREF

  v4 = 0;
  v5 = *(_BYTE *)(a2 + 16);
  v48 = a2;
  if ( v5 != 54 )
  {
    v48 = 0;
    if ( v5 == 55 )
      v4 = a2;
  }
  v50 = sub_1B99570(*(_QWORD *)(a1 + 456), a2, *(_DWORD *)(a1 + 88));
  if ( v50 == 3 )
  {
    sub_1B9E240(a1, a2);
  }
  else
  {
    if ( *(_BYTE *)(a2 + 16) == 54 )
      v6 = *(__int64 **)a2;
    else
      v6 = **(__int64 ***)(a2 - 48);
    v56 = (__int64 **)sub_16463B0(v6, *(_DWORD *)(a1 + 88));
    v7 = (__int64 *)sub_13A4950(a2);
    v51 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
    v8 = sub_15F2050(a2);
    v9 = sub_1632FA0(v8);
    if ( !v51 )
      v51 = sub_15A9FE0(v9, (__int64)v6);
    v55 = sub_1B8DFF0(a2);
    v52 = v50 == 2;
    if ( (unsigned int)(v50 - 1) <= 1 )
    {
      v62[0] = 0;
      v7 = (__int64 *)sub_1B9DCB0(a1, v7, (unsigned int *)v62);
    }
    v60[0] = (unsigned __int64)v61;
    v60[1] = 0x200000000LL;
    v53 = a3 != 0;
    if ( a3 )
      sub_1B8E680((__int64)v60, a3, v10, v11, v12, v13);
    v54 = 0;
    v14 = sub_13A4950(a2);
    v15 = sub_1649C60(v14);
    if ( *(_BYTE *)(v15 + 16) == 56 )
      v54 = sub_15FA300(v15);
    v62[4] = v60;
    v62[0] = &v52;
    v62[2] = &v54;
    v62[3] = &v53;
    v62[5] = &v56;
    v62[1] = a1;
    v62[6] = &v55;
    if ( v4 )
    {
      v16 = 0;
      sub_1B91520(a1, (__int64 *)(a1 + 96), v4);
      if ( *(_DWORD *)(a1 + 92) )
      {
        v49 = v7;
        v46 = (__int64 *)(a1 + 96);
        do
        {
          v17 = (__int64 *)sub_1B9C240((unsigned int *)a1, *(__int64 **)(v4 - 48), v16);
          if ( v50 == 4 )
          {
            v20 = 0;
            if ( v53 )
              v20 = *(_QWORD *)(v60[0] + 8LL * v16);
            v21 = (__int64 *)sub_1B9C240((unsigned int *)a1, v49, v16);
            v19 = sub_15E8270(v46, v17, v21, v51, v20);
          }
          else
          {
            if ( v52 )
              v17 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, v17);
            v18 = (__int64 *)sub_1B982D0((__int64)v62, v16, v49);
            if ( v53 )
            {
              v19 = sub_15E80D0(v46, (__int64)v17, v18, v51, *(_QWORD *)(v60[0] + 8LL * v16));
            }
            else
            {
              v43 = (__int64)v18;
              v59 = 257;
              v31 = sub_1648A60(64, 2u);
              v19 = v31;
              if ( v31 )
                sub_15F9650((__int64)v31, (__int64)v17, v43, 0, 0);
              v32 = *(_QWORD *)(a1 + 104);
              if ( v32 )
              {
                v44 = *(unsigned __int64 **)(a1 + 112);
                sub_157E9D0(v32 + 40, (__int64)v19);
                v33 = *v44;
                v34 = v19[3] & 7LL;
                v19[4] = v44;
                v33 &= 0xFFFFFFFFFFFFFFF8LL;
                v19[3] = v33 | v34;
                *(_QWORD *)(v33 + 8) = v19 + 3;
                *v44 = *v44 & 7 | (unsigned __int64)(v19 + 3);
              }
              sub_164B780((__int64)v19, v58);
              sub_12A86E0(v46, (__int64)v19);
              sub_15F9450((__int64)v19, v51);
            }
          }
          ++v16;
          sub_1B91660(a1, (__int64)v19, v4);
        }
        while ( *(_DWORD *)(a1 + 92) > v16 );
      }
    }
    else
    {
      sub_1B91520(a1, (__int64 *)(a1 + 96), v48);
      if ( *(_DWORD *)(a1 + 92) )
      {
        v42 = (__int64 *)(a1 + 96);
        v22 = 0;
        v47 = (unsigned int *)(a1 + 280);
        do
        {
          if ( v50 == 4 )
          {
            v23 = 0;
            if ( v53 )
              v23 = *(_QWORD *)(v60[0] + 8LL * v22);
            v24 = (__int64 *)sub_1B9C240((unsigned int *)a1, v7, v22);
            v58[0] = (__int64)"wide.masked.gather";
            v59 = 259;
            v57 = sub_15E8160(v42, v24, v51, v23, 0, (__int64)v58);
            sub_1B916B0(a1, (__int64 *)&v57, 1, v48);
            v27 = (__int64)v57;
          }
          else
          {
            v29 = sub_1B982D0((__int64)v62, v22, v7);
            HIBYTE(v59) = 1;
            v40 = (__int64 *)v29;
            if ( v53 )
            {
              LOBYTE(v59) = 3;
              v58[0] = (__int64)"wide.masked.load";
              v30 = sub_1599EF0(v56);
              v57 = sub_15E8010(v42, v40, v51, *(_QWORD *)(v60[0] + 8LL * v22), v30, (__int64)v58);
            }
            else
            {
              LOBYTE(v59) = 3;
              v58[0] = (__int64)"wide.load";
              v35 = sub_1648A60(64, 1u);
              v36 = (__int64)v35;
              if ( v35 )
                sub_15F9210((__int64)v35, *(_QWORD *)(*v40 + 24), (__int64)v40, 0, 0, 0);
              v37 = *(_QWORD *)(a1 + 104);
              if ( v37 )
              {
                v41 = *(__int64 **)(a1 + 112);
                sub_157E9D0(v37 + 40, v36);
                v38 = *v41;
                v39 = *(_QWORD *)(v36 + 24) & 7LL;
                *(_QWORD *)(v36 + 32) = v41;
                v38 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v36 + 24) = v38 | v39;
                *(_QWORD *)(v38 + 8) = v36 + 24;
                *v41 = *v41 & 7 | (v36 + 24);
              }
              sub_164B780(v36, v58);
              sub_12A86E0(v42, v36);
              sub_15F8F50(v36, v51);
              v57 = (_QWORD *)v36;
            }
            sub_1B916B0(a1, (__int64 *)&v57, 1, v48);
            if ( v52 )
              v57 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 32LL))(a1, v57);
            v27 = (__int64)v57;
          }
          v28 = v22++;
          sub_1B99BD0(v47, a2, v28, v27, v25, v26);
        }
        while ( *(_DWORD *)(a1 + 92) > v22 );
      }
    }
    if ( (_BYTE *)v60[0] != v61 )
      _libc_free(v60[0]);
  }
}
