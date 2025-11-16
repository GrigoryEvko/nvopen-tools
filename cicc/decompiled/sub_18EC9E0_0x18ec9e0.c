// Function: sub_18EC9E0
// Address: 0x18ec9e0
//
__int64 __fastcall sub_18EC9E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // r12
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // r8
  unsigned int v13; // esi
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 result; // rax
  char v18; // al
  __int64 v19; // r9
  int v20; // ecx
  unsigned int v21; // edi
  __int64 *v22; // rdx
  __int64 v23; // r10
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // r8
  _QWORD *v27; // rax
  __int64 v28; // rcx
  unsigned __int64 v29; // rdx
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // rsi
  __int64 *v34; // r13
  unsigned int v35; // ecx
  __int64 v36; // rsi
  unsigned __int8 *v37; // rsi
  unsigned int v38; // edx
  __int64 *v39; // r14
  int v40; // esi
  unsigned int v41; // r8d
  __int64 v42; // rdi
  __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rcx
  unsigned __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // r13
  __int64 v50; // r12
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  __int64 v53; // rsi
  unsigned __int8 *v54; // rsi
  int v55; // r11d
  __int64 v56; // rsi
  int v57; // edx
  unsigned int v58; // eax
  __int64 v59; // rcx
  __int64 v60; // rcx
  int v61; // edx
  unsigned int v62; // eax
  __int64 v63; // rsi
  int v64; // r8d
  __int64 *v65; // rdi
  int v66; // edx
  int v67; // edx
  int v68; // r8d
  __int64 v69; // [rsp+8h] [rbp-58h]
  _QWORD *v70; // [rsp+8h] [rbp-58h]
  __int64 v71; // [rsp+8h] [rbp-58h]
  __int64 v72; // [rsp+8h] [rbp-58h]
  __int64 v73; // [rsp+8h] [rbp-58h]
  __int64 v74[2]; // [rsp+10h] [rbp-50h] BYREF
  char v75; // [rsp+20h] [rbp-40h]
  char v76; // [rsp+21h] [rbp-3Fh]

  v6 = a2;
  if ( !a3 )
  {
LABEL_6:
    v10 = *(_QWORD *)a4;
    goto LABEL_7;
  }
  v8 = sub_18E8200(a1, *(_QWORD *)a4, *(_DWORD *)(a4 + 8));
  v76 = 1;
  v74[0] = (__int64)"const_mat";
  v75 = 3;
  v9 = sub_15FB440(11, a2, a3, (__int64)v74, v8);
  v10 = *(_QWORD *)a4;
  v6 = (_QWORD *)v9;
  v11 = *(_QWORD *)(*(_QWORD *)a4 + 48LL);
  v74[0] = v11;
  if ( v11 )
  {
    sub_1623A60((__int64)v74, v11, 2);
    v12 = (__int64)(v6 + 6);
    if ( v6 + 6 == v74 )
    {
      if ( v74[0] )
        sub_161E7C0((__int64)(v6 + 6), v74[0]);
      goto LABEL_6;
    }
    v36 = v6[6];
    if ( !v36 )
    {
LABEL_41:
      v37 = (unsigned __int8 *)v74[0];
      v6[6] = v74[0];
      if ( v37 )
      {
        sub_1623210((__int64)v74, v37, v12);
        v10 = *(_QWORD *)a4;
        v13 = *(_DWORD *)(a4 + 8);
        v14 = *(_QWORD *)a4;
        if ( (*(_BYTE *)(*(_QWORD *)a4 + 23LL) & 0x40) != 0 )
          goto LABEL_8;
        goto LABEL_43;
      }
      goto LABEL_6;
    }
LABEL_40:
    v71 = v12;
    sub_161E7C0(v12, v36);
    v12 = v71;
    goto LABEL_41;
  }
  v12 = v9 + 48;
  if ( (__int64 *)(v9 + 48) != v74 )
  {
    v36 = *(_QWORD *)(v9 + 48);
    if ( v36 )
      goto LABEL_40;
  }
LABEL_7:
  v13 = *(_DWORD *)(a4 + 8);
  v14 = v10;
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
  {
LABEL_8:
    v15 = *(_QWORD *)(v10 - 8);
    goto LABEL_9;
  }
LABEL_43:
  v15 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
LABEL_9:
  v16 = *(_QWORD *)(v15 + 24LL * v13);
  result = *(unsigned __int8 *)(v16 + 16);
  if ( (_BYTE)result == 13 )
  {
    result = sub_18E71D0(v10, v13, (__int64)v6);
    if ( a3 && (_BYTE)result != 1 )
      return sub_15F20C0(v6);
    return result;
  }
  if ( (unsigned __int8)result > 0x17u )
  {
    v18 = *(_BYTE *)(a1 + 64) & 1;
    if ( v18 )
    {
      v19 = a1 + 72;
      v20 = 3;
    }
    else
    {
      v35 = *(_DWORD *)(a1 + 80);
      v19 = *(_QWORD *)(a1 + 72);
      if ( !v35 )
      {
        v38 = *(_DWORD *)(a1 + 64);
        ++*(_QWORD *)(a1 + 56);
        v39 = 0;
        v40 = (v38 >> 1) + 1;
        goto LABEL_48;
      }
      v20 = v35 - 1;
    }
    v21 = v20 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v22 = (__int64 *)(v19 + 16LL * v21);
    v23 = *v22;
    if ( v16 == *v22 )
    {
LABEL_14:
      v24 = v22[1];
      if ( v24 )
        return sub_18E71D0(v14, v13, v24);
      v39 = v22;
LABEL_54:
      v43 = sub_15F4880(v16);
      v39[1] = v43;
      if ( (*(_BYTE *)(v43 + 23) & 0x40) != 0 )
        v44 = *(_QWORD **)(v43 - 8);
      else
        v44 = (_QWORD *)(v43 - 24LL * (*(_DWORD *)(v43 + 20) & 0xFFFFFFF));
      if ( *v44 )
      {
        v45 = v44[1];
        v46 = v44[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v46 = v45;
        if ( v45 )
          *(_QWORD *)(v45 + 16) = *(_QWORD *)(v45 + 16) & 3LL | v46;
      }
      *v44 = v6;
      if ( v6 )
      {
        v47 = v6[1];
        v44[1] = v47;
        if ( v47 )
          *(_QWORD *)(v47 + 16) = (unsigned __int64)(v44 + 1) | *(_QWORD *)(v47 + 16) & 3LL;
        v44[2] = (unsigned __int64)(v6 + 1) | v44[2] & 3LL;
        v6[1] = v44;
      }
      sub_15F2180(v39[1], v16);
      v48 = *(_QWORD *)(v16 + 48);
      v49 = v39[1];
      v74[0] = v48;
      if ( v48 )
      {
        v50 = v49 + 48;
        sub_1623A60((__int64)v74, v48, 2);
        if ( (__int64 *)(v49 + 48) == v74 )
        {
          if ( v74[0] )
            sub_161E7C0(v49 + 48, v74[0]);
          return sub_18E71D0(*(_QWORD *)a4, *(_DWORD *)(a4 + 8), v39[1]);
        }
        v53 = *(_QWORD *)(v49 + 48);
        if ( !v53 )
        {
LABEL_79:
          v54 = (unsigned __int8 *)v74[0];
          *(_QWORD *)(v49 + 48) = v74[0];
          if ( v54 )
            sub_1623210((__int64)v74, v54, v50);
          return sub_18E71D0(*(_QWORD *)a4, *(_DWORD *)(a4 + 8), v39[1]);
        }
      }
      else
      {
        v50 = v49 + 48;
        if ( (__int64 *)(v49 + 48) == v74 )
          return sub_18E71D0(*(_QWORD *)a4, *(_DWORD *)(a4 + 8), v39[1]);
        v53 = *(_QWORD *)(v49 + 48);
        if ( !v53 )
          return sub_18E71D0(*(_QWORD *)a4, *(_DWORD *)(a4 + 8), v39[1]);
      }
      sub_161E7C0(v50, v53);
      goto LABEL_79;
    }
    v55 = 1;
    v39 = 0;
    while ( v23 != -8 )
    {
      if ( v23 == -16 && !v39 )
        v39 = v22;
      v21 = v20 & (v55 + v21);
      v22 = (__int64 *)(v19 + 16LL * v21);
      v23 = *v22;
      if ( v16 == *v22 )
        goto LABEL_14;
      ++v55;
    }
    v41 = 12;
    v35 = 4;
    if ( !v39 )
      v39 = v22;
    v38 = *(_DWORD *)(a1 + 64);
    ++*(_QWORD *)(a1 + 56);
    v40 = (v38 >> 1) + 1;
    if ( v18 )
    {
LABEL_49:
      v42 = a1 + 56;
      if ( 4 * v40 < v41 )
      {
        if ( v35 - *(_DWORD *)(a1 + 68) - v40 > v35 >> 3 )
        {
LABEL_51:
          *(_DWORD *)(a1 + 64) = (2 * (v38 >> 1) + 2) | v38 & 1;
          if ( *v39 != -8 )
            --*(_DWORD *)(a1 + 68);
          *v39 = v16;
          v39[1] = 0;
          goto LABEL_54;
        }
        sub_14163A0(v42, v35);
        if ( (*(_BYTE *)(a1 + 64) & 1) != 0 )
        {
          v60 = a1 + 72;
          v61 = 3;
          goto LABEL_93;
        }
        v67 = *(_DWORD *)(a1 + 80);
        v60 = *(_QWORD *)(a1 + 72);
        if ( v67 )
        {
          v61 = v67 - 1;
LABEL_93:
          v62 = v61 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v39 = (__int64 *)(v60 + 16LL * v62);
          v63 = *v39;
          if ( v16 != *v39 )
          {
            v64 = 1;
            v65 = 0;
            while ( v63 != -8 )
            {
              if ( !v65 && v63 == -16 )
                v65 = v39;
              v62 = v61 & (v64 + v62);
              v39 = (__int64 *)(v60 + 16LL * v62);
              v63 = *v39;
              if ( v16 == *v39 )
                goto LABEL_90;
              ++v64;
            }
LABEL_96:
            if ( v65 )
              v39 = v65;
            goto LABEL_90;
          }
          goto LABEL_90;
        }
LABEL_122:
        *(_DWORD *)(a1 + 64) = (2 * (*(_DWORD *)(a1 + 64) >> 1) + 2) | *(_DWORD *)(a1 + 64) & 1;
        BUG();
      }
      sub_14163A0(v42, 2 * v35);
      if ( (*(_BYTE *)(a1 + 64) & 1) != 0 )
      {
        v56 = a1 + 72;
        v57 = 3;
      }
      else
      {
        v66 = *(_DWORD *)(a1 + 80);
        v56 = *(_QWORD *)(a1 + 72);
        if ( !v66 )
          goto LABEL_122;
        v57 = v66 - 1;
      }
      v58 = v57 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v39 = (__int64 *)(v56 + 16LL * v58);
      v59 = *v39;
      if ( v16 != *v39 )
      {
        v68 = 1;
        v65 = 0;
        while ( v59 != -8 )
        {
          if ( !v65 && v59 == -16 )
            v65 = v39;
          v58 = v57 & (v68 + v58);
          v39 = (__int64 *)(v56 + 16LL * v58);
          v59 = *v39;
          if ( v16 == *v39 )
            goto LABEL_90;
          ++v68;
        }
        goto LABEL_96;
      }
LABEL_90:
      v38 = *(_DWORD *)(a1 + 64);
      goto LABEL_51;
    }
    v35 = *(_DWORD *)(a1 + 80);
LABEL_48:
    v41 = 3 * v35;
    goto LABEL_49;
  }
  if ( (_BYTE)result != 5 )
    return result;
  v25 = sub_1596970(*(_QWORD *)(v15 + 24LL * v13));
  v26 = v25;
  if ( (*(_BYTE *)(v25 + 23) & 0x40) != 0 )
    v27 = *(_QWORD **)(v25 - 8);
  else
    v27 = (_QWORD *)(v25 - 24LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF));
  if ( *v27 )
  {
    v28 = v27[1];
    v29 = v27[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v29 = v28;
    if ( v28 )
      *(_QWORD *)(v28 + 16) = *(_QWORD *)(v28 + 16) & 3LL | v29;
  }
  *v27 = v6;
  if ( v6 )
  {
    v30 = v6[1];
    v27[1] = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = (unsigned __int64)(v27 + 1) | *(_QWORD *)(v30 + 16) & 3LL;
    v27[2] = (unsigned __int64)(v6 + 1) | v27[2] & 3LL;
    v6[1] = v27;
  }
  v69 = v26;
  v31 = sub_18E8200(a1, *(_QWORD *)a4, *(_DWORD *)(a4 + 8));
  sub_15F2120(v69, v31);
  v32 = v69;
  v33 = *(_QWORD *)(*(_QWORD *)a4 + 48LL);
  v34 = (__int64 *)(v69 + 48);
  v74[0] = v33;
  if ( !v33 )
  {
    if ( v34 == v74 )
      goto LABEL_30;
    v51 = *(_QWORD *)(v69 + 48);
    if ( !v51 )
      goto LABEL_30;
LABEL_70:
    v72 = v32;
    sub_161E7C0((__int64)v34, v51);
    v32 = v72;
    goto LABEL_71;
  }
  sub_1623A60((__int64)v74, v33, 2);
  v32 = v69;
  if ( v34 == v74 )
  {
    if ( v74[0] )
    {
      sub_161E7C0((__int64)v34, v74[0]);
      v32 = v69;
    }
    goto LABEL_30;
  }
  v51 = *(_QWORD *)(v69 + 48);
  if ( v51 )
    goto LABEL_70;
LABEL_71:
  v52 = (unsigned __int8 *)v74[0];
  *(_QWORD *)(v32 + 48) = v74[0];
  if ( v52 )
  {
    v73 = v32;
    sub_1623210((__int64)v74, v52, (__int64)v34);
    v32 = v73;
  }
LABEL_30:
  v70 = (_QWORD *)v32;
  result = sub_18E71D0(*(_QWORD *)a4, *(_DWORD *)(a4 + 8), v32);
  if ( !(_BYTE)result )
  {
    result = sub_15F20C0(v70);
    if ( a3 )
      return sub_15F20C0(v6);
  }
  return result;
}
