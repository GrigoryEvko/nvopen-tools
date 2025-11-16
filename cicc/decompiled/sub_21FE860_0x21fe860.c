// Function: sub_21FE860
// Address: 0x21fe860
//
char __fastcall sub_21FE860(_QWORD *a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v5; // rax
  __int16 v6; // dx
  char v7; // al
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r14
  __int64 *v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rsi
  __int64 v15; // r12
  __int64 v16; // r14
  __int32 v17; // eax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 **v23; // rax
  _QWORD *v24; // rax
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // esi
  __int64 v29; // rdi
  unsigned int v30; // ecx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // r11d
  __int64 v36; // r9
  int v37; // ecx
  int v38; // eax
  int v39; // esi
  __int64 v40; // r8
  unsigned int v41; // edx
  __int64 v42; // rdi
  int v43; // r11d
  __int64 v44; // r9
  int v45; // eax
  int v46; // edx
  __int64 v47; // rdi
  __int64 v48; // r8
  unsigned int v49; // ebx
  int v50; // r9d
  __int64 v51; // rsi
  __int64 *v52; // r12
  __int64 *v54; // [rsp+8h] [rbp-B8h]
  __int64 v55; // [rsp+10h] [rbp-B0h]
  unsigned int v59; // [rsp+2Ch] [rbp-94h]
  __int64 *v60; // [rsp+30h] [rbp-90h]
  __int64 v61; // [rsp+38h] [rbp-88h]
  _QWORD *v62; // [rsp+40h] [rbp-80h]
  __int64 v63; // [rsp+50h] [rbp-70h]
  __m128i v64; // [rsp+60h] [rbp-60h] BYREF
  __int64 v65; // [rsp+70h] [rbp-50h]
  __int64 v66; // [rsp+78h] [rbp-48h]
  __int64 v67; // [rsp+80h] [rbp-40h]

  v60 = *(__int64 **)(*(_QWORD *)(a2 + 24) + 56LL);
  v5 = *(_QWORD *)(a2 + 16);
  if ( *(_WORD *)v5 != 1 || (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 8) == 0 )
  {
    v6 = *(_WORD *)(a2 + 46);
    if ( (v6 & 4) != 0 || (v6 & 8) == 0 )
      v7 = WORD1(*(_QWORD *)(v5 + 8)) & 1;
    else
      v7 = sub_1E15D00(a2, 0x10000u, 1);
    if ( !v7 )
    {
      if ( **(_WORD **)(a2 + 16) != 1 || (v8 = *(_QWORD *)(a2 + 32), (*(_BYTE *)(v8 + 64) & 0x10) == 0) )
      {
        LOWORD(v8) = *(_WORD *)(a2 + 46);
        if ( (v8 & 4) == 0 && (v8 & 8) != 0 )
          LOBYTE(v8) = sub_1E15D00(a2, 0x20000u, 1);
      }
      return v8;
    }
  }
  v9 = (_QWORD *)sub_15E0530(*v60);
  switch ( **(_WORD **)(a2 + 16) )
  {
    case 0xB89:
    case 0xB8A:
    case 0xB8B:
    case 0xB8C:
      v10 = 189248;
      v55 = 2;
      v54 = (__int64 *)sub_16432A0(v9);
      v59 = 2;
      break;
    case 0xB8F:
    case 0xB90:
    case 0xB91:
    case 0xB92:
      v10 = 189632;
      v55 = 4;
      v54 = (__int64 *)sub_16432A0(v9);
      v59 = 4;
      break;
    case 0xB95:
    case 0xB96:
    case 0xB97:
    case 0xB98:
      v10 = 190016;
      v55 = 2;
      v54 = (__int64 *)sub_16432B0(v9);
      v59 = 2;
      break;
    case 0xB9B:
    case 0xB9C:
    case 0xB9D:
    case 0xB9E:
      v10 = 190400;
      v55 = 4;
      v54 = (__int64 *)sub_16432B0(v9);
      v59 = 4;
      break;
    case 0xBA1:
    case 0xBA2:
    case 0xBA3:
    case 0xBA4:
      v10 = 190784;
      v55 = 2;
      v54 = (__int64 *)sub_1643340(v9);
      v59 = 2;
      break;
    case 0xBA7:
    case 0xBA8:
    case 0xBA9:
    case 0xBAA:
      v10 = 191168;
      v55 = 4;
      v54 = (__int64 *)sub_1643340(v9);
      v59 = 4;
      break;
    case 0xBAD:
    case 0xBAE:
    case 0xBAF:
    case 0xBB0:
      v10 = 191552;
      v55 = 2;
      v54 = (__int64 *)sub_1643350(v9);
      v59 = 2;
      break;
    case 0xBB3:
    case 0xBB4:
    case 0xBB5:
    case 0xBB6:
      v10 = 191936;
      v55 = 4;
      v54 = (__int64 *)sub_1643350(v9);
      v59 = 4;
      break;
    case 0xBB9:
    case 0xBBA:
    case 0xBBB:
    case 0xBBC:
      v10 = 192320;
      v55 = 2;
      v54 = (__int64 *)sub_1643360(v9);
      v59 = 2;
      break;
    case 0xBBF:
    case 0xBC0:
    case 0xBC1:
    case 0xBC2:
      v10 = 192704;
      v55 = 4;
      v54 = (__int64 *)sub_1643360(v9);
      v59 = 4;
      break;
    case 0xBC5:
    case 0xBC6:
    case 0xBC7:
    case 0xBC8:
      v10 = 193088;
      v55 = 2;
      v54 = (__int64 *)sub_1643330(v9);
      v59 = 2;
      break;
    case 0xBCB:
    case 0xBCC:
    case 0xBCD:
    case 0xBCE:
      v10 = 193472;
      v55 = 4;
      v54 = (__int64 *)sub_1643330(v9);
      v59 = 4;
      break;
    case 0xBDD:
    case 0xBDE:
    case 0xBDF:
    case 0xBE0:
      v10 = 194624;
      v55 = 1;
      v54 = (__int64 *)sub_16432A0(v9);
      v59 = 1;
      break;
    case 0xBE3:
    case 0xBE4:
    case 0xBE5:
    case 0xBE6:
      v10 = 195008;
      v55 = 1;
      v54 = (__int64 *)sub_16432B0(v9);
      v59 = 1;
      break;
    case 0xBE9:
    case 0xBEA:
    case 0xBEB:
    case 0xBEC:
      v10 = 195392;
      v55 = 1;
      v54 = (__int64 *)sub_1643340(v9);
      v59 = 1;
      break;
    case 0xBEF:
    case 0xBF0:
    case 0xBF1:
    case 0xBF2:
      v10 = 195776;
      v55 = 1;
      v54 = (__int64 *)sub_1643350(v9);
      v59 = 1;
      break;
    case 0xBF5:
    case 0xBF6:
    case 0xBF7:
    case 0xBF8:
      v10 = 196160;
      v55 = 1;
      v54 = (__int64 *)sub_1643360(v9);
      v59 = 1;
      break;
    case 0xBFB:
    case 0xBFC:
    case 0xBFD:
    case 0xBFE:
      v10 = 196544;
      v55 = 1;
      v54 = (__int64 *)sub_1643330(v9);
      v59 = 1;
      break;
    default:
      v10 = 0;
      break;
  }
  v11 = *(_QWORD *)(a2 + 24);
  v12 = (__int64 *)(a2 + 64);
  v13 = *(_QWORD *)(v11 + 56);
  v14 = *(_QWORD *)(a1[62] + 8LL) + v10;
  if ( (*(_BYTE *)(a2 + 46) & 4) != 0 )
  {
    v15 = (__int64)sub_1E0B640(v13, v14, v12, 0);
    sub_1DD6E10(v11, (__int64 *)a2, v15);
  }
  else
  {
    v15 = (__int64)sub_1E0B640(v13, v14, v12, 0);
    sub_1DD5BA0((__int64 *)(v11 + 16), v15);
    v32 = *(_QWORD *)a2;
    v33 = *(_QWORD *)v15;
    *(_QWORD *)(v15 + 8) = a2;
    v32 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v15 = v32 | v33 & 7;
    *(_QWORD *)(v32 + 8) = v15;
    *(_QWORD *)a2 = v15 | *(_QWORD *)a2 & 7LL;
  }
  v16 = 0;
  v61 = 40LL * v59;
  do
  {
    v17 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + v16 + 8);
    v64.m128i_i64[0] = 0x10000000;
    v16 += 40;
    v65 = 0;
    v64.m128i_i32[2] = v17;
    v66 = 0;
    v67 = 0;
    sub_1E1A9C0(v15, v13, &v64);
  }
  while ( v61 != v16 );
  v18 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + v61 + 24);
  v64.m128i_i64[0] = 1;
  v65 = 0;
  v66 = v18;
  sub_1E1A9C0(v15, v13, &v64);
  v64.m128i_i64[0] = 1;
  v65 = 0;
  v66 = 4;
  sub_1E1A9C0(v15, v13, &v64);
  v64.m128i_i64[0] = 1;
  v65 = 0;
  v66 = v55;
  sub_1E1A9C0(v15, v13, &v64);
  v19 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (v59 + 3) + 24);
  v64.m128i_i64[0] = 1;
  v65 = 0;
  v66 = v19;
  sub_1E1A9C0(v15, v13, &v64);
  v20 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (v59 + 4) + 24);
  v64.m128i_i64[0] = 1;
  v65 = 0;
  v66 = v20;
  sub_1E1A9C0(v15, v13, &v64);
  v21 = *sub_21D7770(
           a1[63],
           v60,
           858993459 * (unsigned int)((__int64)(*(_QWORD *)(a1[59] + 16LL) - *(_QWORD *)(a1[59] + 8LL)) >> 3) + a3);
  v64.m128i_i8[0] = 9;
  v65 = 0;
  v66 = v21;
  v64.m128i_i32[0] &= 0xFFF000FF;
  v64.m128i_i32[2] = 0;
  LODWORD(v67) = 0;
  sub_1E1A9C0(v15, v13, &v64);
  if ( v59 + 7 == *(_DWORD *)(a2 + 40) )
  {
    v34 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (v59 + 6) + 24);
    v64.m128i_i64[0] = 1;
    v65 = 0;
    v66 = v34;
  }
  else
  {
    v64.m128i_i64[0] = 1;
    v65 = 0;
    v66 = 0;
  }
  sub_1E1A9C0(v15, v13, &v64);
  v22 = **(_QWORD **)(a2 + 56);
  v23 = (__int64 **)sub_1646BA0(v54, 0);
  v24 = (_QWORD *)sub_1599A20(v23);
  LOBYTE(v63) = 0;
  v64 = 0u;
  v65 = 0;
  v62 = v24;
  v25 = 0;
  if ( v24 )
  {
    v26 = *v24;
    if ( *(_BYTE *)(v26 + 8) == 16 )
      v26 = **(_QWORD **)(v26 + 16);
    v25 = *(_DWORD *)(v26 + 8) >> 8;
  }
  HIDWORD(v63) = v25;
  v27 = sub_1E0B8E0(
          (__int64)v60,
          *(_WORD *)(v22 + 32),
          *(_QWORD *)(v22 + 24),
          (unsigned int)(1 << *(_WORD *)(v22 + 34)) >> 1,
          (int)&v64,
          0,
          (unsigned __int64)v62,
          v63,
          1u,
          0,
          0);
  sub_1E15C90(v15, v13, v27);
  v28 = *(_DWORD *)(a4 + 24);
  if ( v28 )
  {
    v29 = *(_QWORD *)(a4 + 8);
    v30 = (v28 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = v29 + 16LL * v30;
    v31 = *(_QWORD *)v8;
    if ( *(_QWORD *)v8 == a2 )
    {
      v59 += *(_DWORD *)(v8 + 8);
      goto LABEL_28;
    }
    v35 = 1;
    v36 = 0;
    while ( v31 != -8 )
    {
      if ( v31 != -16 || v36 )
        v8 = v36;
      v30 = (v28 - 1) & (v35 + v30);
      v52 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v52;
      if ( *v52 == a2 )
      {
        v8 = v29 + 16LL * v30;
        v59 += *((_DWORD *)v52 + 2);
        goto LABEL_28;
      }
      ++v35;
      v36 = v8;
      v8 = v29 + 16LL * v30;
    }
    if ( v36 )
      v8 = v36;
    ++*(_QWORD *)a4;
    v37 = *(_DWORD *)(a4 + 16) + 1;
    if ( 4 * v37 < 3 * v28 )
    {
      if ( v28 - *(_DWORD *)(a4 + 20) - v37 > v28 >> 3 )
        goto LABEL_56;
      sub_1DC6D40(a4, v28);
      v45 = *(_DWORD *)(a4 + 24);
      if ( v45 )
      {
        v46 = v45 - 1;
        v47 = *(_QWORD *)(a4 + 8);
        v48 = 0;
        v49 = (v45 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v50 = 1;
        v37 = *(_DWORD *)(a4 + 16) + 1;
        v8 = v47 + 16LL * v49;
        v51 = *(_QWORD *)v8;
        if ( *(_QWORD *)v8 != a2 )
        {
          while ( v51 != -8 )
          {
            if ( !v48 && v51 == -16 )
              v48 = v8;
            v49 = v46 & (v50 + v49);
            v8 = v47 + 16LL * v49;
            v51 = *(_QWORD *)v8;
            if ( *(_QWORD *)v8 == a2 )
              goto LABEL_56;
            ++v50;
          }
          if ( v48 )
            v8 = v48;
        }
        goto LABEL_56;
      }
LABEL_89:
      ++*(_DWORD *)(a4 + 16);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)a4;
  }
  sub_1DC6D40(a4, 2 * v28);
  v38 = *(_DWORD *)(a4 + 24);
  if ( !v38 )
    goto LABEL_89;
  v39 = v38 - 1;
  v40 = *(_QWORD *)(a4 + 8);
  v41 = (v38 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v37 = *(_DWORD *)(a4 + 16) + 1;
  v8 = v40 + 16LL * v41;
  v42 = *(_QWORD *)v8;
  if ( *(_QWORD *)v8 != a2 )
  {
    v43 = 1;
    v44 = 0;
    while ( v42 != -8 )
    {
      if ( v42 == -16 && !v44 )
        v44 = v8;
      v41 = v39 & (v43 + v41);
      v8 = v40 + 16LL * v41;
      v42 = *(_QWORD *)v8;
      if ( *(_QWORD *)v8 == a2 )
        goto LABEL_56;
      ++v43;
    }
    if ( v44 )
      v8 = v44;
  }
LABEL_56:
  *(_DWORD *)(a4 + 16) = v37;
  if ( *(_QWORD *)v8 != -8 )
    --*(_DWORD *)(a4 + 20);
  *(_QWORD *)v8 = a2;
  *(_DWORD *)(v8 + 8) = 0;
LABEL_28:
  *(_DWORD *)(v8 + 8) = v59;
  return v8;
}
