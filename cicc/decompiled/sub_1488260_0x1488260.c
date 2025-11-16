// Function: sub_1488260
// Address: 0x1488260
//
__int64 __fastcall sub_1488260(__int64 a1, __int64 a2, _QWORD *a3, __m128i a4, __m128i a5)
{
  __int64 *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rdi
  unsigned int v11; // ebx
  int v12; // eax
  __int64 v13; // rax
  __int64 *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 *v17; // rdx
  __int64 *v19; // r12
  __int64 v20; // r12
  __int64 *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // eax
  char v25; // bl
  __int64 v26; // rax
  _BYTE *v27; // rsi
  __int64 v28; // rax
  __int64 *v29; // r15
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // r9
  int v33; // eax
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned int v38; // ebx
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rax
  __int64 *v44; // rax
  __int64 *v45; // r15
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 *v49; // rax
  int v50; // eax
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // r12
  __int64 v54; // rax
  __int64 v55; // [rsp+10h] [rbp-C0h]
  __int64 v56; // [rsp+10h] [rbp-C0h]
  __int64 *v58; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v59; // [rsp+28h] [rbp-A8h]
  __int64 *v60; // [rsp+30h] [rbp-A0h] BYREF
  int v61; // [rsp+38h] [rbp-98h]
  __int64 *v62; // [rsp+40h] [rbp-90h] BYREF
  int v63; // [rsp+48h] [rbp-88h]
  __int64 *v64; // [rsp+50h] [rbp-80h] BYREF
  __int64 *v65; // [rsp+58h] [rbp-78h]
  __int64 v66; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v67; // [rsp+70h] [rbp-60h] BYREF
  __int64 v68; // [rsp+78h] [rbp-58h]
  _BYTE v69[80]; // [rsp+80h] [rbp-50h] BYREF

  v7 = *(__int64 **)(a1 + 32);
  v8 = *(_QWORD *)(a1 + 40);
  v9 = *v7;
  if ( *(_WORD *)(*v7 + 24) )
  {
    v22 = 8 * v8;
    v14 = &v7[(unsigned __int64)v22 / 8];
    v16 = v22 >> 3;
    if ( v22 > 31 || v22 == 16 )
      goto LABEL_12;
    goto LABEL_22;
  }
  v10 = *(_QWORD *)(v9 + 32);
  v11 = *(_DWORD *)(v10 + 32);
  if ( v11 > 0x40 )
  {
    v55 = v8;
    v12 = sub_16A57B0(v10 + 24);
    v8 = v55;
    if ( v11 == v12 )
      goto LABEL_4;
LABEL_15:
    v68 = 0x400000000LL;
    v67 = (__int64 *)v69;
    sub_145C5B0((__int64)&v67, v7, &v7[v8]);
    v19 = v67;
    *v19 = sub_145CF80((__int64)a3, **(_QWORD **)(v9 + 32), 0, 0);
    v20 = sub_14785F0((__int64)a3, &v67, *(_QWORD *)(a1 + 48), *(_WORD *)(a1 + 26) & 1);
    if ( *(_WORD *)(v20 + 24) == 7 )
    {
      sub_158BC30(&v64, a2, *(_QWORD *)(v9 + 32) + 24LL);
      v21 = (__int64 *)sub_1488A30(v20, &v64, a3);
      sub_135E100(&v66);
      sub_135E100((__int64 *)&v64);
      goto LABEL_17;
    }
    goto LABEL_32;
  }
  if ( *(_QWORD *)(v10 + 24) )
    goto LABEL_15;
LABEL_4:
  v13 = 8 * v8;
  v14 = &v7[v8];
  v15 = (8 * v8) >> 5;
  v16 = v13 >> 3;
  if ( v15 <= 0 )
  {
    if ( v13 == 16 )
      goto LABEL_37;
  }
  else
  {
    v17 = &v7[4 * v15];
    while ( 1 )
    {
      if ( *(_WORD *)(v7[1] + 24) )
      {
        ++v7;
        goto LABEL_12;
      }
      if ( *(_WORD *)(v7[2] + 24) )
      {
        v7 += 2;
        goto LABEL_12;
      }
      if ( *(_WORD *)(v7[3] + 24) )
      {
        v7 += 3;
        goto LABEL_12;
      }
      v7 += 4;
      if ( v17 == v7 )
        break;
      if ( *(_WORD *)(*v7 + 24) )
        goto LABEL_12;
    }
    v16 = v14 - v7;
    if ( (char *)v14 - (char *)v7 == 16 )
      goto LABEL_36;
  }
LABEL_22:
  if ( v16 != 3 )
  {
    if ( v16 != 1 )
      goto LABEL_25;
LABEL_24:
    if ( !*(_WORD *)(*v7 + 24) )
      goto LABEL_25;
    goto LABEL_12;
  }
  if ( *(_WORD *)(*v7 + 24) )
    goto LABEL_12;
  ++v7;
LABEL_36:
  if ( !*(_WORD *)(*v7 + 24) )
  {
LABEL_37:
    ++v7;
    goto LABEL_24;
  }
LABEL_12:
  if ( v7 != v14 )
    return sub_1456E90((__int64)a3);
LABEL_25:
  v23 = sub_1456040(v9);
  v24 = sub_1456C90((__int64)a3, v23);
  sub_135E0D0((__int64)&v67, v24, 0, 0);
  v25 = sub_158B950(a2, &v67);
  sub_135E100((__int64 *)&v67);
  if ( v25 )
  {
    v26 = *(_QWORD *)(a1 + 40);
    if ( v26 != 2 )
    {
      if ( v26 == 3 )
      {
        v27 = *(_BYTE **)(a1 + 32);
        v68 = 0x400000000LL;
        v67 = (__int64 *)v69;
        sub_145C5B0((__int64)&v67, v27, v27 + 24);
        v28 = sub_145CF40((__int64)a3, a2 + 16);
        v29 = v67;
        *v29 = sub_1480620((__int64)a3, v28, 0);
        v30 = sub_14785F0((__int64)a3, &v67, *(_QWORD *)(a1 + 48), 0);
        sub_145D730((__int64)&v64, v30, (__int64)a3);
        if ( (_BYTE)v66 )
        {
          v21 = v64;
          v45 = v65;
          v46 = sub_15A35F0(36, v64[4], v65[4], 0, v31, v32);
          if ( *(_BYTE *)(v46 + 16) == 13 )
          {
            if ( *(_DWORD *)(v46 + 32) <= 0x40u )
              v47 = *(_QWORD *)(v46 + 24);
            else
              v47 = **(_QWORD **)(v46 + 24);
            if ( !v47 )
              v21 = v45;
            v48 = sub_145CE20((__int64)a3, v21[4]);
            v49 = sub_1487810(a1, v48, a3, a4, a5);
            if ( (unsigned __int8)sub_158B950(a2, v49[4] + 24) )
            {
              sub_13A38D0((__int64)&v60, v21[4] + 24);
              sub_16A7490(&v60, 1);
              v50 = v61;
              v51 = a3[3];
              v61 = 0;
              v63 = v50;
              v62 = v60;
              v52 = sub_15E0530(v51);
              v53 = sub_159C0E0(v52, &v62);
              sub_135E100((__int64 *)&v62);
              sub_135E100((__int64 *)&v60);
              v54 = sub_1487E60(a1, v53, a3, a4, a5);
              if ( !(unsigned __int8)sub_158B950(a2, v54 + 24) )
              {
                v21 = (__int64 *)sub_145CE20((__int64)a3, v53);
LABEL_17:
                if ( v67 != (__int64 *)v69 )
                  _libc_free((unsigned __int64)v67);
                return (__int64)v21;
              }
            }
            else
            {
              sub_13A38D0((__int64)&v60, v21[4] + 24);
              sub_16A7800(&v60, 1);
              v33 = v61;
              v34 = a3[3];
              v61 = 0;
              v63 = v33;
              v62 = v60;
              v35 = sub_15E0530(v34);
              v56 = sub_159C0E0(v35, &v62);
              sub_135E100((__int64 *)&v62);
              sub_135E100((__int64 *)&v60);
              v36 = sub_1487E60(a1, v56, a3, a4, a5);
              if ( (unsigned __int8)sub_158B950(a2, v36 + 24) )
                goto LABEL_17;
            }
LABEL_32:
            v21 = (__int64 *)sub_1456E90((__int64)a3);
            goto LABEL_17;
          }
        }
        if ( v67 != (__int64 *)v69 )
          _libc_free((unsigned __int64)v67);
      }
      return sub_1456E90((__int64)a3);
    }
    sub_13A38D0((__int64)&v58, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL) + 32LL) + 24LL);
    if ( v59 <= 0x40 )
    {
      v39 = (__int64)((_QWORD)v58 << (64 - (unsigned __int8)v59)) >> (64 - (unsigned __int8)v59);
    }
    else
    {
      v38 = v59 + 1;
      if ( sub_13D0200((__int64 *)&v58, v59 - 1) )
      {
        if ( v38 - (unsigned int)sub_16A5810(&v58) > 0x40 )
          goto LABEL_60;
      }
      else if ( v38 - (unsigned int)sub_1455840((__int64)&v58) > 0x40 )
      {
        goto LABEL_44;
      }
      v39 = *v58;
    }
    if ( v39 > 0 )
    {
LABEL_44:
      sub_13A38D0((__int64)&v67, a2 + 16);
      sub_16A7800(&v67, 1);
      v61 = v68;
      v60 = v67;
LABEL_45:
      sub_13A38D0((__int64)&v64, (__int64)&v60);
      sub_16A7200(&v64, &v58);
      v40 = (int)v65;
      LODWORD(v65) = 0;
      LODWORD(v68) = v40;
      v67 = v64;
      sub_16A9D70(&v62, &v67, &v58);
      sub_135E100((__int64 *)&v67);
      sub_135E100((__int64 *)&v64);
      v41 = sub_15E0530(a3[3]);
      v42 = sub_159C0E0(v41, &v62);
      v43 = sub_145CE20((__int64)a3, v42);
      v44 = sub_1487810(a1, v43, a3, a4, a5);
      if ( (unsigned __int8)sub_158B950(a2, v44[4] + 24) )
        v21 = (__int64 *)sub_1456E90((__int64)a3);
      else
        v21 = (__int64 *)sub_145CE20((__int64)a3, v42);
      sub_135E100((__int64 *)&v62);
      sub_135E100((__int64 *)&v60);
      sub_135E100((__int64 *)&v58);
      return (__int64)v21;
    }
LABEL_60:
    sub_13A38D0((__int64)&v60, a2);
    goto LABEL_45;
  }
  v37 = sub_1456040(**(_QWORD **)(a1 + 32));
  return sub_145CF80((__int64)a3, v37, 0, 0);
}
