// Function: sub_31DCC70
// Address: 0x31dcc70
//
__int64 __fastcall sub_31DCC70(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rdx
  _DWORD *v8; // rax
  _DWORD *i; // rdx
  int v10; // eax
  __int64 v11; // rdx
  _DWORD *v12; // rax
  _DWORD *k; // rdx
  char v14; // r14
  __int64 v15; // rax
  unsigned __int8 v16; // dl
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 result; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 (*v28)(); // rcx
  unsigned int v29; // ecx
  unsigned int v30; // eax
  _DWORD *v31; // rdi
  int v32; // r14d
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdi
  _DWORD *v35; // rax
  __int64 v36; // rdx
  _DWORD *m; // rdx
  unsigned int v38; // ecx
  unsigned int v39; // eax
  _DWORD *v40; // rdi
  int v41; // r14d
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rax
  _DWORD *v44; // rax
  __int64 v45; // rdx
  _DWORD *j; // rdx
  __int64 v47; // rax
  _DWORD *v48; // rax
  _DWORD *v49; // rax
  __int64 v50[4]; // [rsp+0h] [rbp-50h] BYREF
  char v51; // [rsp+20h] [rbp-30h]
  char v52; // [rsp+21h] [rbp-2Fh]

  *(_QWORD *)(a1 + 232) = a2;
  v3 = *a2;
  if ( (unsigned __int8)sub_2E79020(a2) )
  {
    *(_BYTE *)(a1 + 780) = 1;
    v4 = a2[6];
    if ( *(_QWORD *)(v4 + 48) || *(_BYTE *)(v4 + 670) )
    {
      if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 21LL) )
        goto LABEL_31;
LABEL_4:
      v5 = sub_31DB510(a1, *a2);
      *(_QWORD *)(a1 + 280) = v5;
      goto LABEL_5;
    }
  }
  v26 = *(_QWORD *)(a1 + 208);
  *(_BYTE *)(a1 + 781) = 1;
  if ( !*(_BYTE *)(v26 + 21) )
    goto LABEL_4;
LABEL_31:
  v27 = sub_31DA6B0(a1);
  v28 = *(__int64 (**)())(*(_QWORD *)v27 + 256LL);
  v5 = 0;
  if ( v28 == sub_302E4D0 )
  {
    *(_QWORD *)(a1 + 280) = 0;
  }
  else
  {
    v5 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v28)(v27, v3, *(_QWORD *)(a1 + 200));
    *(_QWORD *)(a1 + 280) = v5;
  }
LABEL_5:
  *(_QWORD *)(a1 + 296) = v5;
  v6 = *(_DWORD *)(a1 + 320);
  ++*(_QWORD *)(a1 + 304);
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 324) )
      goto LABEL_11;
    v7 = *(unsigned int *)(a1 + 328);
    if ( (unsigned int)v7 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 312), 12 * v7, 4);
      *(_QWORD *)(a1 + 312) = 0;
      *(_QWORD *)(a1 + 320) = 0;
      *(_DWORD *)(a1 + 328) = 0;
      goto LABEL_11;
    }
    goto LABEL_8;
  }
  v38 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 328);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v38 = 64;
  if ( (unsigned int)v7 <= v38 )
  {
LABEL_8:
    v8 = *(_DWORD **)(a1 + 312);
    for ( i = &v8[3 * v7]; i != v8; *(v8 - 2) = -1 )
    {
      *v8 = 0;
      v8 += 3;
    }
    *(_QWORD *)(a1 + 320) = 0;
    goto LABEL_11;
  }
  v39 = v6 - 1;
  if ( !v39 )
  {
    v40 = *(_DWORD **)(a1 + 312);
    v41 = 64;
LABEL_52:
    sub_C7D6A0((__int64)v40, 12 * v7, 4);
    v42 = ((((((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
             | (4 * v41 / 3u + 1)
             | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4)
           | (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
           | (4 * v41 / 3u + 1)
           | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
           | (4 * v41 / 3u + 1)
           | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4)
         | (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
         | (4 * v41 / 3u + 1)
         | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 16;
    v43 = (v42
         | (((((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
             | (4 * v41 / 3u + 1)
             | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4)
           | (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
           | (4 * v41 / 3u + 1)
           | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
           | (4 * v41 / 3u + 1)
           | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4)
         | (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
         | (4 * v41 / 3u + 1)
         | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 328) = v43;
    v44 = (_DWORD *)sub_C7D670(12 * v43, 4);
    v45 = *(unsigned int *)(a1 + 328);
    *(_QWORD *)(a1 + 320) = 0;
    *(_QWORD *)(a1 + 312) = v44;
    for ( j = &v44[3 * v45]; j != v44; v44 += 3 )
    {
      if ( v44 )
      {
        *v44 = 0;
        v44[1] = -1;
      }
    }
    goto LABEL_11;
  }
  _BitScanReverse(&v39, v39);
  v40 = *(_DWORD **)(a1 + 312);
  v41 = 1 << (33 - (v39 ^ 0x1F));
  if ( v41 < 64 )
    v41 = 64;
  if ( (_DWORD)v7 != v41 )
    goto LABEL_52;
  *(_QWORD *)(a1 + 320) = 0;
  v48 = &v40[3 * v7];
  do
  {
    if ( v40 )
    {
      *v40 = 0;
      v40[1] = -1;
    }
    v40 += 3;
  }
  while ( v48 != v40 );
LABEL_11:
  v10 = *(_DWORD *)(a1 + 424);
  ++*(_QWORD *)(a1 + 408);
  *(_DWORD *)(a1 + 344) = 0;
  if ( !v10 )
  {
    if ( !*(_DWORD *)(a1 + 428) )
      goto LABEL_17;
    v11 = *(unsigned int *)(a1 + 432);
    if ( (unsigned int)v11 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 416), 16LL * (unsigned int)v11, 8);
      *(_QWORD *)(a1 + 416) = 0;
      *(_QWORD *)(a1 + 424) = 0;
      *(_DWORD *)(a1 + 432) = 0;
      goto LABEL_17;
    }
    goto LABEL_14;
  }
  v29 = 4 * v10;
  v11 = *(unsigned int *)(a1 + 432);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v29 = 64;
  if ( v29 >= (unsigned int)v11 )
  {
LABEL_14:
    v12 = *(_DWORD **)(a1 + 416);
    for ( k = &v12[4 * v11]; k != v12; *(v12 - 3) = -1 )
    {
      *v12 = 0;
      v12 += 4;
    }
    *(_QWORD *)(a1 + 424) = 0;
    goto LABEL_17;
  }
  v30 = v10 - 1;
  if ( !v30 )
  {
    v31 = *(_DWORD **)(a1 + 416);
    v32 = 64;
LABEL_40:
    sub_C7D6A0((__int64)v31, 16LL * (unsigned int)v11, 8);
    v33 = ((((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
             | (4 * v32 / 3u + 1)
             | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
         | (4 * v32 / 3u + 1)
         | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 16;
    v34 = (v33
         | (((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
             | (4 * v32 / 3u + 1)
             | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
         | (4 * v32 / 3u + 1)
         | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 432) = v34;
    v35 = (_DWORD *)sub_C7D670(16 * v34, 8);
    v36 = *(unsigned int *)(a1 + 432);
    *(_QWORD *)(a1 + 424) = 0;
    *(_QWORD *)(a1 + 416) = v35;
    for ( m = &v35[4 * v36]; m != v35; v35 += 4 )
    {
      if ( v35 )
      {
        *v35 = 0;
        v35[1] = -1;
      }
    }
    goto LABEL_17;
  }
  _BitScanReverse(&v30, v30);
  v31 = *(_DWORD **)(a1 + 416);
  v32 = 1 << (33 - (v30 ^ 0x1F));
  if ( v32 < 64 )
    v32 = 64;
  if ( v32 != (_DWORD)v11 )
    goto LABEL_40;
  *(_QWORD *)(a1 + 424) = 0;
  v49 = &v31[4 * v32];
  do
  {
    if ( v31 )
    {
      *v31 = 0;
      v31[1] = -1;
    }
    v31 += 4;
  }
  while ( v49 != v31 );
LABEL_17:
  v14 = *(_BYTE *)(*(_QWORD *)(a1 + 208) + 81LL);
  v15 = sub_B92180(*a2);
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 200) + 544LL) - 42) > 1
    || v15
    && ((v16 = *(_BYTE *)(v15 - 16), (v16 & 2) != 0)
      ? (v17 = *(_QWORD *)(v15 - 32))
      : (v17 = v15 - 16 - 8LL * ((v16 >> 2) & 0xF)),
        *(_DWORD *)(*(_QWORD *)(v17 + 40) + 32LL) != 3) )
  {
    if ( (unsigned __int8)sub_B2D620(v3, "patchable-function-entry", 0x18u)
      || (unsigned __int8)sub_B2D620(v3, "function-instrument", 0x13u)
      || (unsigned __int8)sub_B2D620(v3, "xray-instruction-threshold", 0x1Au)
      || (unsigned __int8)sub_31D5EF0((__int64)a2, a1) )
    {
      v52 = 1;
      v50[0] = (__int64)"func_begin";
      v51 = 3;
      v21 = sub_31DCC50(a1, v50, v18, v19, v20);
      *(_QWORD *)(a1 + 536) = v21;
      if ( !v14 )
        goto LABEL_24;
    }
    else
    {
      if ( !v14 )
      {
        v47 = a2[1];
        if ( (*(_BYTE *)(v47 + 878) & 0x40) != 0 || (*(_BYTE *)(v47 + 879) & 0x10) != 0 )
        {
          v52 = 1;
          v50[0] = (__int64)"func_begin";
          v51 = 3;
          *(_QWORD *)(a1 + 536) = sub_31DCC50(a1, v50, v18, v19, v20);
        }
        goto LABEL_24;
      }
      v52 = 1;
      v50[0] = (__int64)"func_begin";
      v51 = 3;
      v21 = sub_31DCC50(a1, v50, v18, v19, v20);
      *(_QWORD *)(a1 + 536) = v21;
    }
    *(_QWORD *)(a1 + 296) = v21;
  }
LABEL_24:
  v22 = *(__int64 **)(a1 + 8);
  v23 = *v22;
  v24 = v22[1];
  if ( v23 == v24 )
LABEL_82:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_50209AC )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_82;
  }
  result = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(
                         *(_QWORD *)(v23 + 8),
                         &unk_50209AC)
                     + 200);
  *(_QWORD *)(a1 + 264) = result;
  return result;
}
