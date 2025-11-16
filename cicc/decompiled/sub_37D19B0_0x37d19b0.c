// Function: sub_37D19B0
// Address: 0x37d19b0
//
__int64 __fastcall sub_37D19B0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 *v4; // r12
  __int64 v5; // rax
  __int64 (__fastcall *v6)(__int64); // rcx
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 result; // rax
  __int64 v9; // rax
  unsigned int v10; // r15d
  bool v11; // zf
  __int64 v12; // rdx
  char *v13; // r12
  char *v14; // r15
  _DWORD *v15; // rdx
  int v16; // r11d
  unsigned int v17; // r8d
  _DWORD *v18; // rax
  int v19; // edi
  _QWORD *v20; // rax
  __int64 v21; // r14
  __int64 v22; // rdi
  int v23; // ecx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r9
  int *v28; // rdx
  int v29; // r8d
  unsigned int *v30; // r13
  unsigned int v31; // eax
  int v32; // esi
  __int64 v33; // r14
  int v34; // ecx
  __int64 v35; // rdi
  int v36; // edx
  int v37; // r11d
  __int64 v38; // r13
  int v39; // r12d
  int v40; // ebx
  unsigned int *v41; // r14
  unsigned int v42; // edx
  __int64 v43; // rsi
  unsigned __int64 v44; // rbx
  _DWORD *v45; // rax
  _DWORD *v46; // r14
  _DWORD *v47; // r12
  __int64 v48; // rdx
  unsigned int v49; // esi
  unsigned int v50; // eax
  unsigned int v52; // [rsp+10h] [rbp-B0h]
  unsigned int v53; // [rsp+14h] [rbp-ACh]
  __int64 v54; // [rsp+18h] [rbp-A8h]
  int v55; // [rsp+24h] [rbp-9Ch] BYREF
  _DWORD *v56; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v57[2]; // [rsp+30h] [rbp-90h] BYREF
  char v58; // [rsp+40h] [rbp-80h]
  _QWORD v59[2]; // [rsp+50h] [rbp-70h] BYREF
  char v60; // [rsp+60h] [rbp-60h]
  __int64 v61; // [rsp+70h] [rbp-50h] BYREF
  _DWORD *v62; // [rsp+78h] [rbp-48h]
  __int64 v63; // [rsp+80h] [rbp-40h]
  unsigned int v64; // [rsp+88h] [rbp-38h]

  v2 = a2;
  if ( *(_WORD *)(a2 + 68) == 20 )
  {
    v9 = *(_QWORD *)(a2 + 32);
    v54 = v9 + 40;
  }
  else
  {
    v4 = *(__int64 **)(a1 + 32);
    v5 = *v4;
    v6 = *(__int64 (__fastcall **)(__int64))(*v4 + 520);
    if ( v6 == sub_2DCA430 )
    {
LABEL_3:
      v7 = *(__int64 (__fastcall **)(__int64))(v5 + 528);
      if ( v7 == sub_2E77FE0 )
        return 0;
      ((void (__fastcall *)(_QWORD *, __int64 *, unsigned __int64))v7)(v57, v4, a2);
      v9 = v57[0];
      v54 = v57[1];
      if ( !v58 )
        return 0;
      goto LABEL_7;
    }
    ((void (__fastcall *)(_QWORD *, __int64 *, unsigned __int64))v6)(v59, v4, a2);
    v9 = v59[0];
    v54 = v59[1];
    if ( !v60 )
    {
      v5 = *v4;
      goto LABEL_3;
    }
  }
LABEL_7:
  v10 = *(_DWORD *)(v9 + 8);
  result = 1;
  v53 = *(_DWORD *)(v54 + 8);
  if ( v53 == v10 )
    return result;
  if ( (_BYTE)qword_50512E8
    && (!(unsigned __int8)sub_37BB3A0(a1, v10)
     || (_BYTE)qword_50512E8 && (((*(_BYTE *)(v54 + 3) & 0x40) != 0) & ((*(_BYTE *)(v54 + 3) >> 4) ^ 1)) == 0) )
  {
    return 0;
  }
  v11 = *(_QWORD *)(a1 + 432) == 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  if ( v11 )
    goto LABEL_28;
  v13 = sub_E922F0(*(_QWORD **)(a1 + 16), v10);
  if ( v13 == &v13[2 * v12] )
    goto LABEL_28;
  v52 = v10;
  v14 = &v13[2 * v12];
  do
  {
    v21 = *(_QWORD *)(a1 + 408);
    v22 = *(_QWORD *)(v21 + 64);
    v23 = *(_DWORD *)(v22 + 4LL * *(unsigned __int16 *)v13);
    v24 = *(_QWORD *)(a1 + 432);
    v25 = *(_QWORD *)(v24 + 3416);
    v26 = *(unsigned int *)(v24 + 3432);
    v55 = v23;
    if ( (_DWORD)v26 )
    {
      LODWORD(v27) = (v26 - 1) & v23;
      v28 = (int *)(v25 + 88LL * (unsigned int)v27);
      v29 = *v28;
      if ( v23 == *v28 )
      {
LABEL_18:
        if ( v28 != (int *)(v25 + 88 * v26) && (v28[4] || *((_QWORD *)v28 + 10)) )
        {
          v30 = (unsigned int *)(v22 + 4LL * *(unsigned __int16 *)v13);
          v31 = *v30;
          if ( *v30 == -1 )
          {
            v31 = sub_37BA230(v21, *(unsigned __int16 *)v13);
            *v30 = v31;
          }
          v32 = v64;
          v33 = *(_QWORD *)(*(_QWORD *)(v21 + 32) + 8LL * v31);
          if ( !v64 )
          {
            ++v61;
            v56 = 0;
            goto LABEL_25;
          }
          v15 = 0;
          v16 = 1;
          v17 = (v64 - 1) & v55;
          v18 = &v62[4 * v17];
          v19 = *v18;
          if ( v55 == *v18 )
          {
LABEL_13:
            v20 = v18 + 2;
LABEL_14:
            *v20 = v33;
            goto LABEL_15;
          }
          while ( v19 != -1 )
          {
            if ( v19 == -2 && !v15 )
              v15 = v18;
            v17 = (v64 - 1) & (v16 + v17);
            v18 = &v62[4 * v17];
            v19 = *v18;
            if ( v55 == *v18 )
              goto LABEL_13;
            ++v16;
          }
          if ( !v15 )
            v15 = v18;
          ++v61;
          v34 = v63 + 1;
          v56 = v15;
          if ( 4 * ((int)v63 + 1) >= 3 * v64 )
          {
LABEL_25:
            v32 = 2 * v64;
          }
          else if ( v64 - HIDWORD(v63) - v34 > v64 >> 3 )
          {
            goto LABEL_60;
          }
          sub_37C7C70((__int64)&v61, v32);
          sub_37BE4A0((__int64)&v61, &v55, &v56);
          v15 = v56;
          v34 = v63 + 1;
LABEL_60:
          LODWORD(v63) = v34;
          if ( *v15 != -1 )
            --HIDWORD(v63);
          *v15 = v55;
          *((_QWORD *)v15 + 1) = unk_5051170;
          v20 = v15 + 2;
          goto LABEL_14;
        }
      }
      else
      {
        v36 = 1;
        while ( v29 != -1 )
        {
          v37 = v36 + 1;
          v27 = ((_DWORD)v26 - 1) & (unsigned int)(v27 + v36);
          v28 = (int *)(v25 + 88 * v27);
          v29 = *v28;
          if ( v23 == *v28 )
            goto LABEL_18;
          v36 = v37;
        }
      }
    }
LABEL_15:
    v13 += 2;
  }
  while ( v14 != v13 );
  v10 = v52;
  v2 = a2;
LABEL_28:
  sub_37BB480(a1, v53, v10);
  v35 = *(_QWORD *)(a1 + 432);
  if ( v35 )
  {
    if ( !(_DWORD)v63 )
      goto LABEL_30;
    v45 = v62;
    v46 = &v62[4 * v64];
    if ( v62 == v46 )
      goto LABEL_30;
    while ( 1 )
    {
      v47 = v45;
      if ( *v45 <= 0xFFFFFFFD )
        break;
      v45 += 4;
      if ( v46 == v45 )
        goto LABEL_30;
    }
    if ( v45 == v46 )
      goto LABEL_30;
    while ( 1 )
    {
      v48 = *((_QWORD *)v47 + 1);
      v49 = *v47;
      v47 += 4;
      sub_37CFFC0(v35, v49, v48, v2, 0);
      if ( v47 == v46 )
        break;
      while ( *v47 > 0xFFFFFFFD )
      {
        v47 += 4;
        if ( v46 == v47 )
          goto LABEL_71;
      }
      v35 = *(_QWORD *)(a1 + 432);
      if ( v46 == v47 )
        goto LABEL_72;
    }
LABEL_71:
    v35 = *(_QWORD *)(a1 + 432);
LABEL_72:
    if ( v35 )
    {
LABEL_30:
      if ( (unsigned __int8)sub_37BB3A0(a1, v10)
        && (((*(_BYTE *)(v54 + 3) & 0x40) != 0) & ((*(_BYTE *)(v54 + 3) >> 4) ^ 1)) != 0 )
      {
        sub_37CEB50(
          *(_QWORD *)(a1 + 432),
          *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 408) + 64LL) + 4LL * v53),
          *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 408) + 64LL) + 4LL * v10),
          v2);
      }
    }
  }
  if ( (_BYTE)qword_50512E8 )
  {
    v38 = *(_QWORD *)(a1 + 408);
    v39 = *(_DWORD *)(a1 + 420);
    v40 = *(_DWORD *)(a1 + 416);
    v41 = (unsigned int *)(*(_QWORD *)(v38 + 64) + 4LL * v53);
    v42 = *v41;
    if ( *v41 == -1 )
    {
      v50 = sub_37BA230(v38, v53);
      *v41 = v50;
      v42 = v50;
    }
    v43 = *(_QWORD *)(v38 + 32) + 8LL * v42;
    v44 = *(_QWORD *)v43 & 0xFFFFFF0000000000LL | v40 & 0xFFFFF | ((unsigned __int64)(v39 & 0xFFFFF) << 20);
    *(_QWORD *)v43 = v44;
    *(_DWORD *)(v43 + 4) = BYTE4(v44) | (v42 << 8);
  }
  sub_C7D6A0((__int64)v62, 16LL * v64, 8);
  return 1;
}
