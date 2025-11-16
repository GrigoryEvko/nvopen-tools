// Function: sub_2F44460
// Address: 0x2f44460
//
int *__fastcall sub_2F44460(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, unsigned __int8 a5, __int64 a6)
{
  __int64 v8; // r15
  _QWORD *v9; // rbx
  __int64 v10; // r8
  unsigned __int16 ***v11; // r13
  __int64 v12; // rax
  __int64 v13; // r10
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // r15d
  __int64 v17; // rbx
  unsigned __int16 **v18; // r10
  unsigned int v19; // ebx
  __int64 v20; // r14
  unsigned __int16 *v21; // r11
  unsigned __int16 *v22; // rax
  unsigned __int16 *v23; // r14
  int v24; // r11d
  unsigned __int16 *v25; // r10
  unsigned int v26; // eax
  unsigned int v27; // eax
  unsigned int v28; // ebx
  unsigned __int16 v29; // r15
  _QWORD *v30; // rdi
  _QWORD *v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned int v34; // r9d
  __int16 *v35; // rdi
  __int16 *v36; // rdx
  unsigned int v37; // ecx
  int v38; // esi
  int v39; // eax
  _DWORD *v40; // rax
  __int64 i; // rsi
  unsigned __int64 v42; // rax
  _DWORD *v43; // rax
  int v44; // edx
  __int64 v45; // rdi
  int v46; // r9d
  __int64 v47; // rsi
  unsigned int v48; // ecx
  int v49; // r9d
  int v50; // edx
  int *result; // rax
  __int64 v52; // rdi
  __int64 v53; // r9
  unsigned int v54; // edx
  __int64 v55; // rsi
  int v56; // r9d
  int v57; // r9d
  __int64 v58; // rax
  int v59; // edx
  _QWORD *v60; // rdi
  __int64 v61; // rsi
  _QWORD *v62; // rax
  __int64 v63; // rdx
  __int16 *v64; // rdx
  int v65; // esi
  _QWORD *v66; // r15
  __int64 v67; // rax
  __int64 v68; // r11
  __int64 v69; // rax
  int v70; // edx
  _QWORD *v71; // rdi
  _QWORD *v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // rdi
  __int16 *v75; // rdx
  int v76; // esi
  int v77; // r9d
  int v78; // r9d
  int v79; // r10d
  unsigned int v80; // r9d
  int v81; // r10d
  unsigned int v84; // [rsp+18h] [rbp-48h]
  unsigned int v85; // [rsp+20h] [rbp-40h]
  unsigned int v86; // [rsp+20h] [rbp-40h]
  int v87; // [rsp+20h] [rbp-40h]
  unsigned __int16 v89; // [rsp+2Ah] [rbp-36h]
  unsigned int v90; // [rsp+2Ch] [rbp-34h]

  v8 = *(unsigned int *)(a3 + 8);
  v9 = *(_QWORD **)(a1 + 8);
  v85 = a4;
  v10 = 16 * (v8 & 0x7FFFFFFF);
  v90 = a4;
  v11 = (unsigned __int16 ***)(*(_QWORD *)(v9[7] + v10) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (unsigned int)(a4 - 1) > 0x3FFFFFFE )
  {
    v90 = 0;
    if ( (int)v8 >= 0 )
    {
LABEL_5:
      v14 = *(_QWORD *)(v9[38] + 8 * v8);
      goto LABEL_6;
    }
  }
  else
  {
    v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v9 + 16LL) + 200LL))(*(_QWORD *)(*v9 + 16LL));
    v13 = v90;
    v10 = 16 * (v8 & 0x7FFFFFFF);
    a6 = v85;
    if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v12 + 248) + 16LL) + v90) )
      goto LABEL_3;
    a4 = v90;
    if ( (*(_QWORD *)(v9[48] + 8LL * (v90 >> 6)) & (1LL << v90)) != 0 )
      goto LABEL_3;
    a4 = *((unsigned __int16 *)*v11 + 11);
    v58 = v90 >> 3;
    if ( (unsigned int)v58 >= (unsigned int)a4 )
      goto LABEL_3;
    v59 = *((unsigned __int8 *)(*v11)[1] + v58);
    if ( _bittest(&v59, v90 & 7)
      && (!a5
       || (v60 = *(_QWORD **)(a1 + 1176),
           v61 = (__int64)&v60[*(unsigned int *)(a1 + 1184)],
           v62 = sub_2F413E0(v60, v61, v90),
           v10 = 16 * (v8 & 0x7FFFFFFF),
           a6 = v85,
           (_QWORD *)v61 == v62)) )
    {
      v63 = *(_QWORD *)(a1 + 16);
      a4 = *(_DWORD *)(*(_QWORD *)(v63 + 8) + 24 * v13 + 16) & 0xFFF;
      v64 = (__int16 *)(*(_QWORD *)(v63 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v63 + 8) + 24 * v13 + 16) >> 12));
      do
      {
        if ( !v64 )
          break;
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 1112) + 4LL * (unsigned int)a4) >= (a5 ^ 1u | *(_DWORD *)(a1 + 1104)) )
          goto LABEL_3;
        v65 = *v64++;
        a4 = (unsigned int)(v65 + a4);
      }
      while ( (_WORD)v65 );
      v45 = a1;
      if ( (unsigned __int8)sub_2F422A0(a1, a6) )
      {
        v79 = *(_DWORD *)(a3 + 8);
        *(_WORD *)(a3 + 12) = a6;
        sub_2F42240(a1, a6, v79);
        v47 = (__int64)a2;
        v48 = v80;
        v50 = v81;
        return sub_2F438D0(v45, v47, v50, v48);
      }
      v9 = *(_QWORD **)(a1 + 8);
    }
    else
    {
LABEL_3:
      v90 = 0;
      v9 = *(_QWORD **)(a1 + 8);
    }
    if ( (int)v8 >= 0 )
      goto LABEL_5;
  }
  v14 = *(_QWORD *)(v9[7] + v10 + 8);
LABEL_6:
  if ( !v14 )
    goto LABEL_13;
  if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
  {
    v14 = *(_QWORD *)(v14 + 32);
    if ( !v14 || (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
      goto LABEL_13;
  }
  v15 = *(_QWORD *)(v14 + 16);
  v16 = 3;
  v17 = v14;
  if ( *(_WORD *)(v15 + 68) != 20 )
    goto LABEL_9;
LABEL_39:
  v40 = *(_DWORD **)(v15 + 32);
  if ( (*v40 & 0xFFF00) != 0 || (v40[10] & 0xFFF00) != 0 )
  {
LABEL_9:
    if ( --v16 )
    {
      v14 = *(_QWORD *)(v17 + 16);
      while ( 1 )
      {
        v17 = *(_QWORD *)(v17 + 32);
        if ( !v17 || (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
          break;
        v15 = *(_QWORD *)(v17 + 16);
        if ( v14 != v15 )
        {
          if ( *(_WORD *)(v15 + 68) != 20 )
            goto LABEL_9;
          goto LABEL_39;
        }
      }
    }
    goto LABEL_13;
  }
  v87 = 4;
  for ( i = (unsigned int)v40[12]; (unsigned int)(i - 1) > 0x3FFFFFFE; i = (unsigned int)v43[12] )
  {
    v42 = sub_2EBEE90(*(_QWORD *)(a1 + 8), i);
    if ( !v42 )
      goto LABEL_9;
    if ( *(_WORD *)(v42 + 68) != 20 )
      goto LABEL_9;
    v43 = *(_DWORD **)(v42 + 32);
    if ( (*v43 & 0xFFF00) != 0 )
      goto LABEL_9;
    if ( (v43[10] & 0xFFF00) != 0 )
      goto LABEL_9;
    if ( !--v87 )
      goto LABEL_9;
  }
  v66 = *(_QWORD **)(a1 + 8);
  v19 = i;
  v67 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64))(**(_QWORD **)(*v66 + 16LL)
                                                                                       + 200LL))(
          *(_QWORD *)(*v66 + 16LL),
          i,
          v14,
          a4,
          v10,
          a6);
  v68 = (unsigned int)i;
  if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v67 + 248) + 16LL) + (unsigned int)i) )
  {
LABEL_13:
    v18 = *v11;
    v19 = 0;
    goto LABEL_14;
  }
  a4 = (unsigned int)i;
  v18 = *v11;
  if ( (*(_QWORD *)(v66[48] + 8LL * ((unsigned int)i >> 6)) & (1LL << i)) != 0
    || (v69 = (unsigned int)i >> 3, (unsigned int)v69 >= *((unsigned __int16 *)v18 + 11))
    || (v70 = *((unsigned __int8 *)v18[1] + v69), !_bittest(&v70, i & 7))
    || a5
    && (v71 = *(_QWORD **)(a1 + 1176),
        v72 = &v71[*(unsigned int *)(a1 + 1184)],
        v72 != sub_2F413E0(v71, (__int64)v72, v19)) )
  {
LABEL_90:
    v19 = 0;
  }
  else
  {
    v73 = *(_QWORD *)(a1 + 16);
    v74 = *(_QWORD *)(v73 + 8);
    a4 = *(_DWORD *)(v74 + 24 * v68 + 16) & 0xFFF;
    v75 = (__int16 *)(*(_QWORD *)(v73 + 56) + 2LL * (*(_DWORD *)(v74 + 24 * v68 + 16) >> 12));
    do
    {
      if ( !v75 )
        break;
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 1112) + 4LL * (unsigned int)a4) >= (a5 ^ 1u | *(_DWORD *)(a1 + 1104)) )
        goto LABEL_90;
      v76 = *v75++;
      a4 = (unsigned int)(v76 + a4);
    }
    while ( (_WORD)v76 );
    v45 = a1;
    if ( (unsigned __int8)sub_2F422A0(a1, v19) )
    {
      v77 = *(_DWORD *)(a3 + 8);
      *(_WORD *)(a3 + 12) = v19;
      sub_2F42240(a1, v19, v77);
      v47 = (__int64)a2;
      v48 = v19;
      v50 = v78;
      return sub_2F438D0(v45, v47, v50, v48);
    }
  }
LABEL_14:
  v20 = *(_QWORD *)(a1 + 32) + 24LL * *((unsigned __int16 *)v18 + 12);
  if ( *(_DWORD *)(a1 + 40) != *(_DWORD *)v20 )
    sub_2F60630(a1 + 32, v11, 3LL * *((unsigned __int16 *)v18 + 12), a4);
  v21 = *(unsigned __int16 **)(v20 + 16);
  v86 = -1;
  v89 = 0;
  v22 = &v21[*(unsigned int *)(v20 + 4)];
  if ( v22 != v21 )
  {
    v23 = *(unsigned __int16 **)(v20 + 16);
    v24 = a5 ^ 1;
    v84 = v19;
    v25 = v22;
    do
    {
      v28 = *v23;
      v29 = *v23;
      if ( !a5
        || (v30 = *(_QWORD **)(a1 + 1176),
            v31 = &v30[*(unsigned int *)(a1 + 1184)],
            v31 == sub_2F413E0(v30, (__int64)v31, *v23)) )
      {
        v32 = *(_QWORD *)(a1 + 16);
        v33 = *(_QWORD *)(v32 + 8);
        v34 = *(_DWORD *)(v33 + 24LL * v29 + 16) & 0xFFF;
        v35 = (__int16 *)(*(_QWORD *)(v32 + 56) + 2LL * (*(_DWORD *)(v33 + 24LL * v29 + 16) >> 12));
        v36 = v35;
        v37 = v34;
        do
        {
          if ( !v36 )
            goto LABEL_31;
          if ( *(_DWORD *)(*(_QWORD *)(a1 + 1112) + 4LL * v37) >= ((unsigned int)v24 | *(_DWORD *)(a1 + 1104)) )
            goto LABEL_24;
          v38 = *v36++;
          v37 += v38;
        }
        while ( (_WORD)v38 );
        while ( 1 )
        {
LABEL_31:
          if ( !v35 )
          {
LABEL_50:
            v45 = a1;
            v46 = *(_DWORD *)(a3 + 8);
            *(_WORD *)(a3 + 12) = v29;
            sub_2F42240(a1, v28, v46);
            v47 = (__int64)a2;
            v48 = v28;
            v50 = v49;
            return sub_2F438D0(v45, v47, v50, v48);
          }
          v39 = *(_DWORD *)(*(_QWORD *)(a1 + 808) + 4LL * v34);
          if ( v39 )
            break;
          v44 = *v35++;
          v34 += v44;
          if ( !(_WORD)v44 )
            goto LABEL_50;
        }
        if ( v39 != 1 )
        {
          v26 = v39 & 0x7FFFFFFF;
          if ( *(_DWORD *)(*(_QWORD *)(a1 + 392) + 4LL * v26) != -1 )
            goto LABEL_19;
          v52 = *(unsigned int *)(a1 + 424);
          v53 = *(_QWORD *)(a1 + 416);
          v54 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 624) + 2LL * v26);
          if ( v54 < (unsigned int)v52 )
          {
            while ( 1 )
            {
              v55 = v53 + 24LL * v54;
              if ( v26 == (*(_DWORD *)(v55 + 8) & 0x7FFFFFFF) )
                break;
              v54 += 0x10000;
              if ( (unsigned int)v52 <= v54 )
                goto LABEL_91;
            }
            if ( *(_BYTE *)(v55 + 14) )
              goto LABEL_19;
LABEL_58:
            v27 = 100;
          }
          else
          {
LABEL_91:
            if ( !*(_BYTE *)(v53 + 24 * v52 + 14) )
              goto LABEL_58;
LABEL_19:
            v27 = 50;
          }
          if ( v28 == v84 || v90 == v28 )
            v27 -= 20;
LABEL_22:
          if ( v27 < v86 )
          {
            v86 = v27;
            v89 = v29;
          }
          goto LABEL_24;
        }
        if ( v90 == v28 || v28 == v84 )
        {
          v27 = -21;
          goto LABEL_22;
        }
      }
LABEL_24:
      ++v23;
    }
    while ( v25 != v23 );
    if ( v89 )
    {
      sub_2F424E0(a1, a2, v89);
      v45 = a1;
      v56 = *(_DWORD *)(a3 + 8);
      *(_WORD *)(a3 + 12) = v89;
      sub_2F42240(a1, v89, v56);
      v48 = v89;
      v50 = v57;
      v47 = (__int64)a2;
      return sub_2F438D0(v45, v47, v50, v48);
    }
  }
  result = (int *)sub_2F418E0(a1, a3, (__int64)a2, v11);
  *(_BYTE *)(a3 + 16) = 1;
  *(_WORD *)(a3 + 12) = (_WORD)result;
  return result;
}
