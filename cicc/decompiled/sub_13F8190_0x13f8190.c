// Function: sub_13F8190
// Address: 0x13f8190
//
__int64 __fastcall sub_13F8190(unsigned __int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v6; // r12d
  __int64 v8; // r8
  _QWORD *v9; // rax
  __int64 v10; // r15
  unsigned __int64 v11; // rax
  char v12; // al
  __int64 v13; // rdx
  unsigned int v14; // r12d
  unsigned int v16; // eax
  unsigned int v17; // eax
  unsigned __int64 v18; // r15
  _QWORD *v19; // rax
  __int64 v20; // r8
  __int64 v21; // r15
  unsigned int v22; // eax
  __int64 v23; // r9
  __int64 v24; // rsi
  unsigned __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rsi
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // [rsp+0h] [rbp-1B0h]
  __int64 v33; // [rsp+10h] [rbp-1A0h]
  __int64 v34; // [rsp+18h] [rbp-198h]
  __int64 v35; // [rsp+18h] [rbp-198h]
  __int64 v36; // [rsp+18h] [rbp-198h]
  __int64 v37; // [rsp+20h] [rbp-190h]
  unsigned __int64 v38; // [rsp+20h] [rbp-190h]
  __int64 v39; // [rsp+28h] [rbp-188h]
  __int64 v40; // [rsp+28h] [rbp-188h]
  __int64 v41; // [rsp+28h] [rbp-188h]
  __int64 v42; // [rsp+30h] [rbp-180h]
  __int64 v43; // [rsp+30h] [rbp-180h]
  __int64 v44; // [rsp+30h] [rbp-180h]
  __int64 v45; // [rsp+30h] [rbp-180h]
  unsigned __int64 v47; // [rsp+40h] [rbp-170h] BYREF
  unsigned int v48; // [rsp+48h] [rbp-168h]
  __int64 v49; // [rsp+50h] [rbp-160h] BYREF
  _BYTE *v50; // [rsp+58h] [rbp-158h]
  _BYTE *v51; // [rsp+60h] [rbp-150h]
  __int64 v52; // [rsp+68h] [rbp-148h]
  int v53; // [rsp+70h] [rbp-140h]
  _BYTE v54[312]; // [rsp+78h] [rbp-138h] BYREF

  v6 = a2;
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD **)(*(_QWORD *)a1 + 16LL);
  v10 = *v9;
  if ( !a2 )
  {
    v43 = *(_QWORD *)a1;
    v16 = sub_15A9FE0(a3, *v9);
    v8 = v43;
    v6 = v16;
  }
  v11 = *(unsigned __int8 *)(v10 + 8);
  if ( (unsigned __int8)v11 <= 0xFu )
  {
    v13 = 35454;
    if ( _bittest64(&v13, v11) )
      goto LABEL_10;
  }
  if ( (unsigned int)(v11 - 13) > 1 )
  {
LABEL_5:
    if ( (_DWORD)v11 != 16 )
      return 0;
  }
  v42 = v8;
  v12 = sub_16435F0(v10, 0);
  v8 = v42;
  if ( !v12 )
    return 0;
LABEL_10:
  v49 = 0;
  v50 = v54;
  v51 = v54;
  v52 = 32;
  v53 = 0;
  LODWORD(v11) = *(unsigned __int8 *)(v10 + 8);
  switch ( (int)v13 )
  {
    case 0:
      v44 = v8;
      v19 = (_QWORD *)sub_15A9930(a3, v10);
      v20 = v44;
      v21 = 8LL * *v19;
      break;
    case 1:
      v37 = v8;
      v39 = *(_QWORD *)(v10 + 24);
      v45 = *(_QWORD *)(v10 + 32);
      v22 = sub_15A9FE0(a3, v39);
      v20 = v37;
      v23 = 1;
      v24 = v39;
      v25 = v22;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v24 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v31 = *(_QWORD *)(v24 + 32);
            v24 = *(_QWORD *)(v24 + 24);
            v23 *= v31;
            continue;
          case 1:
            v26 = 16;
            goto LABEL_23;
          case 2:
            v26 = 32;
            goto LABEL_23;
          case 3:
          case 9:
            v26 = 64;
            goto LABEL_23;
          case 4:
            v26 = 80;
            goto LABEL_23;
          case 5:
          case 6:
            v26 = 128;
            goto LABEL_23;
          case 7:
            v36 = v23;
            v29 = 0;
            v41 = v37;
            goto LABEL_30;
          case 0xB:
            v26 = *(_DWORD *)(v24 + 8) >> 8;
            goto LABEL_23;
          case 0xD:
            v35 = v23;
            v28 = (_QWORD *)sub_15A9930(a3, v24);
            v20 = v37;
            v23 = v35;
            v26 = 8LL * *v28;
            goto LABEL_23;
          case 0xE:
            v32 = v23;
            v33 = v37;
            v34 = *(_QWORD *)(v24 + 24);
            v40 = *(_QWORD *)(v24 + 32);
            v38 = (unsigned int)sub_15A9FE0(a3, v34);
            v27 = sub_127FA20(a3, v34);
            v20 = v33;
            v23 = v32;
            v26 = 8 * v38 * v40 * ((v38 + ((unsigned __int64)(v27 + 7) >> 3) - 1) / v38);
            goto LABEL_23;
          case 0xF:
            v36 = v23;
            v41 = v37;
            v29 = *(_DWORD *)(v24 + 8) >> 8;
LABEL_30:
            v30 = sub_15A9520(a3, v29);
            v20 = v41;
            v23 = v36;
            v26 = (unsigned int)(8 * v30);
LABEL_23:
            v21 = 8 * v45 * v25 * ((v25 + ((unsigned __int64)(v26 * v23 + 7) >> 3) - 1) / v25);
            break;
        }
        break;
      }
      break;
    default:
      goto LABEL_5;
  }
  v17 = sub_15A95F0(a3, v20);
  v18 = (unsigned __int64)(v21 + 7) >> 3;
  v48 = v17;
  if ( v17 > 0x40 )
    sub_16A4EF0(&v47, v18, 0);
  else
    v47 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v17) & v18;
  v14 = sub_13F7530(a1, v6, (unsigned __int64)&v47, a3, a4, a5, (__int64)&v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( v51 != v50 )
    _libc_free((unsigned __int64)v51);
  return v14;
}
