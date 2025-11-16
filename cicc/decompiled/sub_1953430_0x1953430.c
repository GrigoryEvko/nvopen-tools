// Function: sub_1953430
// Address: 0x1953430
//
__int64 __fastcall sub_1953430(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  __int64 v4; // r12
  unsigned int v5; // r11d
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  _QWORD *v17; // r11
  __int64 v18; // r8
  int v19; // eax
  _QWORD *v20; // rax
  __int64 v21; // r9
  bool v22; // r11
  __int64 v23; // rbx
  __int64 v24; // r10
  __int64 v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // r8
  __int64 v28; // r10
  bool v29; // r11
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rcx
  _QWORD *v33; // r14
  unsigned __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 i; // r13
  __int64 v39; // rdi
  char v40; // si
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rcx
  _QWORD *v45; // [rsp+18h] [rbp-88h]
  __int64 v46; // [rsp+18h] [rbp-88h]
  __int64 v47; // [rsp+20h] [rbp-80h]
  int v48; // [rsp+20h] [rbp-80h]
  bool v49; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+20h] [rbp-80h]
  __int64 v51; // [rsp+28h] [rbp-78h]
  unsigned __int64 v52; // [rsp+28h] [rbp-78h]
  __int64 v53; // [rsp+28h] [rbp-78h]
  __int64 v54; // [rsp+28h] [rbp-78h]
  bool v55; // [rsp+28h] [rbp-78h]
  __int64 v56; // [rsp+30h] [rbp-70h]
  __int64 v57; // [rsp+30h] [rbp-70h]
  __int64 v58; // [rsp+30h] [rbp-70h]
  __int64 v59; // [rsp+30h] [rbp-70h]
  __int64 v60; // [rsp+38h] [rbp-68h]
  __int64 v61; // [rsp+38h] [rbp-68h]
  unsigned __int8 v62; // [rsp+38h] [rbp-68h]
  _QWORD *v64; // [rsp+48h] [rbp-58h]
  __int64 v65; // [rsp+48h] [rbp-58h]
  __int64 v66; // [rsp+48h] [rbp-58h]
  __int64 v67[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v68; // [rsp+60h] [rbp-40h]
  unsigned __int64 v69; // [rsp+68h] [rbp-38h]

  v3 = sub_157EBA0(a3);
  v4 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v3 + 16) != 26 )
    return 0;
  v5 = 0;
  if ( *(_BYTE *)(v4 + 16) != 77 )
    return v5;
  if ( (*(_DWORD *)(v3 + 20) & 0xFFFFFFF) != 3 || a3 != *(_QWORD *)(v4 + 40) || (*(_DWORD *)(v4 + 20) & 0xFFFFFFF) == 0 )
    return 0;
  v7 = 0;
  v56 = *(_QWORD *)(a2 - 24);
  v60 = a3;
  v8 = 8LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
  while ( 1 )
  {
    v13 = v7 + 24LL * *(unsigned int *)(v4 + 56) + 8;
    if ( (*(_BYTE *)(v4 + 23) & 0x40) == 0 )
      break;
    v9 = *(_QWORD *)(v4 - 8);
    v10 = 3 * v7;
    v11 = *(_QWORD *)(v9 + 3 * v7);
    v12 = *(_QWORD *)(v9 + v13);
    if ( *(_BYTE *)(v11 + 16) == 79 )
      goto LABEL_13;
LABEL_10:
    v7 += 8;
    if ( v8 == v7 )
      return 0;
  }
  v10 = 3 * v7;
  v14 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
  v11 = *(_QWORD *)(v14 + 3 * v7);
  v12 = *(_QWORD *)(v14 + v13);
  if ( *(_BYTE *)(v11 + 16) != 79 )
    goto LABEL_10;
LABEL_13:
  if ( *(_QWORD *)(v11 + 40) != v12 )
    goto LABEL_10;
  v15 = *(_QWORD *)(v11 + 8);
  if ( !v15 )
    goto LABEL_10;
  if ( *(_QWORD *)(v15 + 8) )
    goto LABEL_10;
  v51 = v12;
  v16 = sub_157EBA0(v12);
  if ( *(_BYTE *)(v16 + 16) != 26 || (*(_DWORD *)(v16 + 20) & 0xFFFFFFF) != 1 )
    goto LABEL_10;
  v47 = v51;
  v52 = v16;
  if ( sub_15CD740(*(_QWORD *)(a1 + 24)) )
  {
    sub_13EBC00(*(__int64 **)(a1 + 8));
    v17 = (_QWORD *)v52;
    v18 = v47;
  }
  else
  {
    sub_13EBC50(*(__int64 **)(a1 + 8));
    v18 = v47;
    v17 = (_QWORD *)v52;
  }
  v53 = v18;
  v45 = v17;
  v48 = sub_13F3340(*(__int64 **)(a1 + 8), *(_WORD *)(a2 + 18) & 0x7FFF, *(_QWORD *)(v11 - 48), v56, v18, a3, a2);
  v19 = sub_13F3340(*(__int64 **)(a1 + 8), *(_WORD *)(a2 + 18) & 0x7FFF, *(_QWORD *)(v11 - 24), v56, v53, a3, a2);
  if ( v48 == v19 || (v19 & v48) == -1 )
    goto LABEL_10;
  v64 = v45;
  v46 = v53;
  v57 = v60;
  v49 = v48 != v19 && (v19 & v48) != -1;
  v54 = *(_QWORD *)(v60 + 56);
  v67[0] = (__int64)"select.unfold";
  LOWORD(v68) = 259;
  v61 = sub_157E9C0(v60);
  v20 = (_QWORD *)sub_22077B0(64);
  v21 = v57;
  v22 = v49;
  v23 = (__int64)v20;
  v24 = v46;
  if ( v20 )
  {
    sub_157FB60(v20, v61, (__int64)v67, v54, v57);
    v22 = v49;
    v24 = v46;
    v21 = v57;
  }
  v50 = v21;
  v55 = v22;
  v58 = v24;
  sub_15F2070(v64);
  sub_157E9D0(v23 + 40, (__int64)v64);
  v25 = *(_QWORD *)(v23 + 40);
  v64[4] = v23 + 40;
  v25 &= 0xFFFFFFFFFFFFFFF8LL;
  v64[3] = v25 | v64[3] & 7LL;
  *(_QWORD *)(v25 + 8) = v64 + 3;
  *(_QWORD *)(v23 + 40) = *(_QWORD *)(v23 + 40) & 7LL | (unsigned __int64)(v64 + 3);
  v65 = *(_QWORD *)(v11 - 72);
  v26 = sub_1648A60(56, 3u);
  v28 = v58;
  v29 = v55;
  v30 = v50;
  if ( v26 )
  {
    sub_15F8650((__int64)v26, v23, v50, v65, v58);
    v29 = v55;
    v28 = v58;
    v30 = v50;
  }
  v31 = *(_QWORD *)(v11 - 24);
  if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
    v32 = *(_QWORD *)(v4 - 8);
  else
    v32 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
  v33 = (_QWORD *)(v32 + v10);
  if ( *v33 )
  {
    v32 = v33[1];
    v34 = v33[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v34 = v32;
    if ( v32 )
      *(_QWORD *)(v32 + 16) = v34 | *(_QWORD *)(v32 + 16) & 3LL;
  }
  *v33 = v31;
  if ( v31 )
  {
    v32 = *(_QWORD *)(v31 + 8);
    v33[1] = v32;
    if ( v32 )
      *(_QWORD *)(v32 + 16) = (unsigned __int64)(v33 + 1) | *(_QWORD *)(v32 + 16) & 3LL;
    v33[2] = (v31 + 8) | v33[2] & 3LL;
    *(_QWORD *)(v31 + 8) = v33;
  }
  v62 = v29;
  v59 = v30;
  v66 = v28;
  sub_1704F80(v4, *(_QWORD *)(v11 - 48), v23, v32, v27, v30);
  sub_15F20C0((_QWORD *)v11);
  v67[0] = v23;
  v35 = *(_QWORD *)(a1 + 24);
  v68 = v66;
  v67[1] = v59 & 0xFFFFFFFFFFFFFFFBLL;
  v69 = v23 & 0xFFFFFFFFFFFFFFFBLL;
  sub_15CD9D0(v35, v67, 2);
  v37 = v59;
  for ( i = *(_QWORD *)(v59 + 48); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v39 = i - 24;
    if ( v4 != i - 24 )
    {
      v40 = *(_BYTE *)(i - 1) & 0x40;
      v41 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
      if ( v41 )
      {
        v36 = v39 - 24LL * v41;
        v42 = 0;
        v43 = 24LL * *(unsigned int *)(i + 32) + 8;
        while ( 1 )
        {
          v37 = v39 - 24LL * v41;
          if ( v40 )
            v37 = *(_QWORD *)(i - 32);
          if ( *(_QWORD *)(v37 + v43) == v66 )
            break;
          v42 = (unsigned int)(v42 + 1);
          v43 += 8;
          if ( v41 == (_DWORD)v42 )
            goto LABEL_52;
        }
      }
      else
      {
LABEL_52:
        v42 = 0xFFFFFFFFLL;
      }
      if ( v40 )
        v44 = *(_QWORD *)(i - 32);
      else
        v44 = v39 - 24LL * v41;
      sub_1704F80(v39, *(_QWORD *)(v44 + 24 * v42), v23, v44, v36, v37);
    }
  }
  return v62;
}
