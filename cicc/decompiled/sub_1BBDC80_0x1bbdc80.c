// Function: sub_1BBDC80
// Address: 0x1bbdc80
//
__int64 __fastcall sub_1BBDC80(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, unsigned int *a5, unsigned int *a6)
{
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  bool v17; // cf
  bool v18; // zf
  __int64 result; // rax
  int v20; // r13d
  bool v21; // r11
  bool v22; // r10
  __int64 *v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx
  unsigned __int8 v28; // di
  char v29; // si
  __int64 v30; // rdi
  char v31; // cl
  char v32; // si
  __int64 v33; // r15
  __int64 v34; // rcx
  __int64 v35; // r14
  __int64 v36; // r10
  __int64 v37; // r13
  unsigned int v38; // r14d
  unsigned int v39; // r9d
  __int64 v40; // rdi
  __int64 v41; // r15
  __int64 v42; // rdi
  __int64 v43; // rsi
  char v44; // al
  __int64 v45; // rsi
  __int64 *v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rax
  int v49; // r9d
  bool v50; // [rsp+7h] [rbp-79h]
  bool v51; // [rsp+7h] [rbp-79h]
  bool v52; // [rsp+7h] [rbp-79h]
  const void *v54; // [rsp+10h] [rbp-70h]
  int v55; // [rsp+18h] [rbp-68h]
  bool v56; // [rsp+20h] [rbp-60h]
  bool v57; // [rsp+20h] [rbp-60h]
  bool v58; // [rsp+20h] [rbp-60h]
  bool v59; // [rsp+20h] [rbp-60h]
  __int64 v60; // [rsp+28h] [rbp-58h]
  int v61; // [rsp+28h] [rbp-58h]
  bool v62; // [rsp+28h] [rbp-58h]
  __int64 v63; // [rsp+30h] [rbp-50h]
  __int64 v64; // [rsp+30h] [rbp-50h]
  __int64 v65; // [rsp+30h] [rbp-50h]
  int v66; // [rsp+30h] [rbp-50h]
  bool v68; // [rsp+44h] [rbp-3Ch]
  unsigned int v69; // [rsp+44h] [rbp-3Ch]
  unsigned int v70; // [rsp+44h] [rbp-3Ch]
  bool v71; // [rsp+48h] [rbp-38h]
  __int64 v72; // [rsp+48h] [rbp-38h]
  __int64 v73; // [rsp+48h] [rbp-38h]

  v55 = a4;
  if ( !a4 )
  {
    v15 = *(_QWORD *)a5;
    v14 = *(__int64 **)a6;
    v68 = *(_BYTE *)(**(_QWORD **)a5 + 16LL) > 0x17u;
    v71 = *(_BYTE *)(**(_QWORD **)a6 + 16LL) > 0x17u;
LABEL_14:
    v20 = 0;
    v21 = 1;
    v22 = 1;
    v54 = a6 + 4;
    while ( 1 )
    {
      v33 = (unsigned int)(v20 + 1);
      v49 = v20 + 1;
      v34 = a3[v33];
      v48 = a5[2];
      v23 = (*(_BYTE *)(v34 + 23) & 0x40) != 0
          ? *(__int64 **)(v34 - 8)
          : (__int64 *)(v34 - 24LL * (*(_DWORD *)(v34 + 20) & 0xFFFFFFF));
      v35 = *v23;
      v24 = v23[3];
      if ( v21 )
      {
        v25 = v14[v20];
        if ( v24 == v25 )
          goto LABEL_34;
        if ( v25 == v35 )
          break;
      }
      if ( v22 )
      {
        v26 = *(_QWORD *)(v15 + 8LL * v20);
        if ( v26 == v35 )
          goto LABEL_34;
        if ( v24 == v26 )
          goto LABEL_55;
      }
      v27 = 0;
      v28 = *(_BYTE *)(v24 + 16);
      if ( *(_BYTE *)(v35 + 16) > 0x17u )
        v27 = v35;
      if ( v28 <= 0x17u )
      {
        if ( !v71 )
        {
          v30 = 0;
LABEL_60:
          if ( !v68 )
            goto LABEL_34;
          v32 = *(_BYTE *)(*(_QWORD *)(v15 + 8LL * v20) + 16LL);
          if ( v27 )
          {
            v31 = *(_BYTE *)(v27 + 16);
LABEL_31:
            if ( v31 == v32 )
              goto LABEL_34;
          }
LABEL_32:
          if ( !v30 || *(_BYTE *)(v30 + 16) != v32 )
            goto LABEL_34;
LABEL_55:
          if ( (unsigned int)v48 >= a5[3] )
            goto LABEL_83;
          goto LABEL_56;
        }
        v30 = 0;
        v60 = v20;
        v29 = *(_BYTE *)(v14[v60] + 16);
        if ( !v27 )
          goto LABEL_65;
      }
      else
      {
        if ( !v71 )
        {
          v30 = v24;
          goto LABEL_60;
        }
        v60 = v20;
        v29 = *(_BYTE *)(v14[v60] + 16);
        if ( v28 == v29 )
          goto LABEL_34;
        v30 = v24;
        if ( !v27 )
        {
LABEL_65:
          if ( !v68 )
            goto LABEL_34;
          v32 = *(_BYTE *)(*(_QWORD *)(v15 + v60 * 8) + 16LL);
          goto LABEL_32;
        }
      }
      v31 = *(_BYTE *)(v27 + 16);
      if ( v29 != v31 )
      {
        if ( !v68 )
          goto LABEL_34;
        v32 = *(_BYTE *)(*(_QWORD *)(v15 + v60 * 8) + 16LL);
        goto LABEL_31;
      }
      if ( !v68 )
        goto LABEL_55;
      if ( v29 == *(_BYTE *)(*(_QWORD *)(v15 + v60 * 8) + 16LL) )
        goto LABEL_34;
      if ( (unsigned int)v48 >= a5[3] )
      {
LABEL_83:
        v50 = v21;
        v56 = v22;
        v63 = v24;
        sub_16CD150((__int64)a5, a5 + 4, 0, 8, v24, v49);
        v15 = *(_QWORD *)a5;
        v48 = a5[2];
        v21 = v50;
        v22 = v56;
        v49 = v20 + 1;
        v24 = v63;
      }
LABEL_56:
      *(_QWORD *)(v15 + 8 * v48) = v24;
      ++a5[2];
      result = a6[2];
      if ( (unsigned int)result >= a6[3] )
      {
        v59 = v21;
        v62 = v22;
        v66 = v49;
        sub_16CD150((__int64)a6, v54, 0, 8, v24, v49);
        result = a6[2];
        v21 = v59;
        v22 = v62;
        v49 = v66;
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * result) = v35;
      ++a6[2];
LABEL_39:
      if ( v21 )
      {
        result = *(_QWORD *)a6;
        v21 = *(_QWORD *)(*(_QWORD *)a6 + 8LL * (unsigned int)v20) == *(_QWORD *)(*(_QWORD *)a6 + 8 * v33);
      }
      if ( v22 )
      {
        result = *(_QWORD *)a5;
        v22 = *(_QWORD *)(*(_QWORD *)a5 + 8LL * (unsigned int)v20) == *(_QWORD *)(*(_QWORD *)a5 + 8 * v33);
      }
      if ( v68 )
      {
        v68 = 0;
        result = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a5 + 8 * v33) + 16LL);
        if ( (unsigned __int8)result > 0x17u )
          v68 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a5 + 8LL * (unsigned int)v20) + 16LL) == (unsigned __int8)result;
      }
      if ( v71 )
      {
        v71 = 0;
        result = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a6 + 8 * v33) + 16LL);
        if ( (unsigned __int8)result > 0x17u )
          v71 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a6 + 8LL * (unsigned int)v20) + 16LL) == (unsigned __int8)result;
      }
      if ( v20 == v55 - 2 )
      {
        if ( v21 || v22 )
          return result;
        v36 = a1;
        v37 = 8;
        v38 = 0;
        v39 = v55 - 1;
        while ( 2 )
        {
          ++v38;
          v41 = v37 - 8;
          v42 = *(_QWORD *)(*(_QWORD *)a5 + v37 - 8);
          result = *(_QWORD *)a6;
          if ( *(_BYTE *)(v42 + 16) == 54 && (v43 = *(_QWORD *)(result + v37), *(_BYTE *)(v43 + 16) == 54) )
          {
            v69 = v39;
            v72 = v36;
            v44 = sub_385F290(v42, v43, *(_QWORD *)(v36 + 1376), *(_QWORD *)(v36 + 1312), 1);
            v36 = v72;
            v39 = v69;
            if ( !v44 )
            {
              result = *(_QWORD *)a6;
              v40 = *(_QWORD *)(*(_QWORD *)a6 + v41);
              if ( *(_BYTE *)(v40 + 16) == 54 )
                goto LABEL_76;
              goto LABEL_71;
            }
          }
          else
          {
            v40 = *(_QWORD *)(result + v41);
            if ( *(_BYTE *)(v40 + 16) != 54 )
              goto LABEL_71;
LABEL_76:
            result = *(_QWORD *)a5;
            v45 = *(_QWORD *)(*(_QWORD *)a5 + v37);
            if ( *(_BYTE *)(v45 + 16) != 54
              || (v70 = v39,
                  v73 = v36,
                  result = sub_385F290(v40, v45, *(_QWORD *)(v36 + 1376), *(_QWORD *)(v36 + 1312), 1),
                  v36 = v73,
                  v39 = v70,
                  !(_BYTE)result) )
            {
LABEL_71:
              v37 += 8;
              if ( v39 <= v38 )
                return result;
              continue;
            }
          }
          break;
        }
        result = v37 + *(_QWORD *)a6;
        v46 = (__int64 *)(v37 + *(_QWORD *)a5);
        v47 = *v46;
        *v46 = *(_QWORD *)result;
        *(_QWORD *)result = v47;
        goto LABEL_71;
      }
      v15 = *(_QWORD *)a5;
      v14 = *(__int64 **)a6;
      v20 = v49;
    }
    if ( v22 && *(_QWORD *)(v15 + 8LL * v20) == v35 )
    {
LABEL_34:
      if ( (unsigned int)v48 >= a5[3] )
      {
        v52 = v21;
        v58 = v22;
        v65 = v24;
        sub_16CD150((__int64)a5, a5 + 4, 0, 8, v24, v49);
        v15 = *(_QWORD *)a5;
        v48 = a5[2];
        v21 = v52;
        v22 = v58;
        v49 = v20 + 1;
        v24 = v65;
      }
      *(_QWORD *)(v15 + 8 * v48) = v35;
      ++a5[2];
      result = a6[2];
      if ( (unsigned int)result >= a6[3] )
      {
        v51 = v21;
        v57 = v22;
        v61 = v49;
        v64 = v24;
        sub_16CD150((__int64)a6, v54, 0, 8, v24, v49);
        result = a6[2];
        v21 = v51;
        v22 = v57;
        v49 = v61;
        v24 = v64;
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * result) = v24;
      ++a6[2];
      goto LABEL_39;
    }
    goto LABEL_55;
  }
  v8 = *a3;
  if ( (*(_BYTE *)(*a3 + 23) & 0x40) != 0 )
    v9 = *(__int64 **)(v8 - 8);
  else
    v9 = (__int64 *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
  v10 = *v9;
  v11 = v9[3];
  if ( *(_BYTE *)(v11 + 16) <= 0x17u && *(_BYTE *)(v10 + 16) > 0x17u )
  {
    v10 = v9[3];
    v11 = *v9;
  }
  v12 = a5[2];
  if ( (unsigned int)v12 >= a5[3] )
  {
    sub_16CD150((__int64)a5, a5 + 4, 0, 8, (int)a5, (int)a6);
    v12 = a5[2];
  }
  *(_QWORD *)(*(_QWORD *)a5 + 8 * v12) = v10;
  ++a5[2];
  v13 = a6[2];
  if ( (unsigned int)v13 >= a6[3] )
  {
    sub_16CD150((__int64)a6, a6 + 4, 0, 8, (int)a5, (int)a6);
    v13 = a6[2];
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v13) = v11;
  v14 = *(__int64 **)a6;
  ++a6[2];
  v15 = *(_QWORD *)a5;
  v16 = **(_QWORD **)a5;
  v17 = *(_BYTE *)(v16 + 16) < 0x17u;
  v18 = *(_BYTE *)(v16 + 16) == 23;
  result = *v14;
  v68 = !v17 && !v18;
  v71 = *(_BYTE *)(*v14 + 16) > 0x17u;
  if ( v55 != 1 )
    goto LABEL_14;
  return result;
}
