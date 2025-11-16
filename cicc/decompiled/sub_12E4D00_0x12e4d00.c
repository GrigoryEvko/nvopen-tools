// Function: sub_12E4D00
// Address: 0x12e4d00
//
__int64 __fastcall sub_12E4D00(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned int v19; // esi
  _QWORD *v20; // r12
  int v21; // edx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r8
  unsigned int v27; // edx
  _QWORD *v28; // rax
  __int64 v29; // rdi
  int v30; // r9d
  int v31; // r10d
  int v32; // eax
  _QWORD v33[2]; // [rsp+0h] [rbp-140h] BYREF
  __int64 v34; // [rsp+10h] [rbp-130h]
  __int64 v35; // [rsp+20h] [rbp-120h]
  _QWORD v36[2]; // [rsp+28h] [rbp-118h] BYREF
  __int64 v37; // [rsp+38h] [rbp-108h]
  void *v38; // [rsp+40h] [rbp-100h]
  _QWORD v39[2]; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v40; // [rsp+58h] [rbp-E8h]
  __int64 v41; // [rsp+60h] [rbp-E0h]
  void *v42; // [rsp+70h] [rbp-D0h]
  _QWORD v43[2]; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v44; // [rsp+88h] [rbp-B8h]
  __int64 v45; // [rsp+90h] [rbp-B0h]
  _QWORD *v46; // [rsp+A0h] [rbp-A0h] BYREF
  _QWORD v47[2]; // [rsp+A8h] [rbp-98h] BYREF
  __int64 v48; // [rsp+B8h] [rbp-88h]
  __int64 v49; // [rsp+C0h] [rbp-80h]
  void *v50; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v51; // [rsp+D8h] [rbp-68h] BYREF
  __int64 v52; // [rsp+E0h] [rbp-60h]
  __int64 v53; // [rsp+E8h] [rbp-58h]
  __int64 v54; // [rsp+F0h] [rbp-50h]
  _QWORD v55[2]; // [rsp+F8h] [rbp-48h] BYREF
  __int64 v56; // [rsp+108h] [rbp-38h]

  v3 = a1[1];
  v39[1] = 0;
  v39[0] = v3 & 6;
  v40 = a1[3];
  result = v40;
  if ( v40 != 0 && v40 != -8 && v40 != -16 )
  {
    sub_1649AC0(v39, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v40;
  }
  v5 = a1[4];
  v41 = v5;
  v38 = &unk_49E6B50;
  v6 = *(unsigned int *)(v5 + 24);
  if ( !(_DWORD)v6 )
    goto LABEL_5;
  v7 = *(_QWORD *)(v5 + 8);
  v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v9 = (_QWORD *)(v7 + ((unsigned __int64)v8 << 6));
  v10 = v9[3];
  if ( v10 == result )
  {
LABEL_10:
    if ( v9 == (_QWORD *)(v7 + (v6 << 6)) )
      goto LABEL_5;
    v33[0] = 6;
    v11 = v9[7];
    v33[1] = 0;
    v34 = v11;
    if ( v11 != -8 && v11 != 0 && v11 != -16 )
    {
      sub_1649AC0(v33, v9[5] & 0xFFFFFFFFFFFFFFF8LL);
      v12 = v9[7];
      v5 = v41;
      if ( v12 != 0 && v12 != -8 && v12 != -16 )
        sub_1649B30(v9 + 5);
    }
    v51 = 2;
    v52 = 0;
    v53 = -16;
    v50 = &unk_49E6B50;
    v54 = 0;
    v13 = v9[3];
    if ( v13 == -16 )
    {
      v9[4] = 0;
    }
    else
    {
      if ( v13 == -8 || !v13 )
      {
        v9[3] = -16;
      }
      else
      {
        sub_1649B30(v9 + 1);
        v14 = v53;
        v15 = v53 == -8;
        v9[3] = v53;
        if ( v14 == 0 || v15 || v14 == -16 )
        {
          v9[4] = v54;
          goto LABEL_24;
        }
        sub_1649AC0(v9 + 1, v51 & 0xFFFFFFFFFFFFFFF8LL);
      }
      v16 = v53;
      v15 = v53 == -8;
      v9[4] = v54;
      v50 = &unk_49EE2B0;
      if ( v16 != -16 && v16 != 0 && !v15 )
        sub_1649B30(&v51);
    }
LABEL_24:
    --*(_DWORD *)(v5 + 16);
    ++*(_DWORD *)(v5 + 20);
    v35 = a2;
    v17 = v41;
    v37 = v34;
    v36[0] = 6;
    v36[1] = 0;
    if ( v34 != -8 && v34 != 0 && v34 != -16 )
    {
      sub_1649AC0(v36, v33[0] & 0xFFFFFFFFFFFFFFF8LL);
      a2 = v35;
    }
    v44 = a2;
    v43[0] = 2;
    v43[1] = 0;
    if ( a2 == 0 || a2 == -8 || a2 == -16 )
    {
      v45 = v17;
      v42 = &unk_49E6B50;
      v18 = v17;
      v51 = 2;
      v52 = 0;
      v53 = a2;
    }
    else
    {
      sub_164C220(v43);
      v42 = &unk_49E6B50;
      v45 = v17;
      v52 = 0;
      v51 = v43[0] & 6;
      v53 = v44;
      if ( v44 == -8 || v44 == 0 || v44 == -16 )
      {
        v18 = v17;
      }
      else
      {
        sub_1649AC0(&v51, v43[0] & 0xFFFFFFFFFFFFFFF8LL);
        v18 = v45;
      }
    }
    v54 = v18;
    v50 = &unk_49E6B50;
    v55[0] = 6;
    v55[1] = 0;
    v56 = v37;
    if ( v37 != 0 && v37 != -8 && v37 != -16 )
      sub_1649AC0(v55, v36[0] & 0xFFFFFFFFFFFFFFF8LL);
    v19 = *(_DWORD *)(v17 + 24);
    if ( v19 )
    {
      v26 = *(_QWORD *)(v17 + 8);
      v27 = (v19 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v28 = (_QWORD *)(v26 + ((unsigned __int64)v27 << 6));
      v29 = v28[3];
      if ( v53 == v29 )
      {
LABEL_56:
        if ( v56 != 0 && v56 != -8 && v56 != -16 )
          sub_1649B30(v55);
        v50 = &unk_49EE2B0;
        if ( v53 != 0 && v53 != -8 && v53 != -16 )
          sub_1649B30(&v51);
        v42 = &unk_49EE2B0;
        if ( v44 != 0 && v44 != -8 && v44 != -16 )
          sub_1649B30(v43);
        if ( v37 != 0 && v37 != -8 && v37 != -16 )
          sub_1649B30(v36);
        if ( v34 != 0 && v34 != -8 && v34 != -16 )
          sub_1649B30(v33);
        result = v40;
        goto LABEL_5;
      }
      v31 = 1;
      v20 = 0;
      while ( v29 != -8 )
      {
        if ( v20 || v29 != -16 )
          v28 = v20;
        v27 = (v19 - 1) & (v31 + v27);
        v29 = *(_QWORD *)(v26 + ((unsigned __int64)v27 << 6) + 24);
        if ( v53 == v29 )
          goto LABEL_56;
        ++v31;
        v20 = v28;
        v28 = (_QWORD *)(v26 + ((unsigned __int64)v27 << 6));
      }
      if ( !v20 )
        v20 = v28;
      v32 = *(_DWORD *)(v17 + 16);
      ++*(_QWORD *)v17;
      v21 = v32 + 1;
      if ( 4 * (v32 + 1) < 3 * v19 )
      {
        if ( v19 - *(_DWORD *)(v17 + 20) - v21 > v19 >> 3 )
        {
LABEL_39:
          *(_DWORD *)(v17 + 16) = v21;
          v47[0] = 2;
          v47[1] = 0;
          v48 = -8;
          v49 = 0;
          if ( v20[3] != -8 )
          {
            --*(_DWORD *)(v17 + 20);
            v46 = &unk_49EE2B0;
            if ( v48 != -8 && v48 != 0 && v48 != -16 )
              sub_1649B30(v47);
          }
          v22 = v20[3];
          v23 = v53;
          if ( v22 != v53 )
          {
            if ( v22 != 0 && v22 != -8 && v22 != -16 )
            {
              sub_1649B30(v20 + 1);
              v23 = v53;
            }
            v20[3] = v23;
            if ( v23 != 0 && v23 != -8 && v23 != -16 )
              sub_1649AC0(v20 + 1, v51 & 0xFFFFFFFFFFFFFFF8LL);
          }
          v24 = v54;
          v20[5] = 6;
          v20[6] = 0;
          v20[4] = v24;
          v25 = v56;
          v15 = v56 == -8;
          v20[7] = v56;
          if ( v25 != 0 && !v15 && v25 != -16 )
            sub_1649AC0(v20 + 5, v55[0] & 0xFFFFFFFFFFFFFFF8LL);
          goto LABEL_56;
        }
LABEL_38:
        sub_12E48B0(v17, v19);
        sub_12E4800(v17, (__int64)&v50, &v46);
        v20 = v46;
        v21 = *(_DWORD *)(v17 + 16) + 1;
        goto LABEL_39;
      }
    }
    else
    {
      ++*(_QWORD *)v17;
    }
    v19 *= 2;
    goto LABEL_38;
  }
  v30 = 1;
  while ( v10 != -8 )
  {
    v8 = (v6 - 1) & (v30 + v8);
    v9 = (_QWORD *)(v7 + ((unsigned __int64)v8 << 6));
    v10 = v9[3];
    if ( v10 == result )
      goto LABEL_10;
    ++v30;
  }
LABEL_5:
  v38 = &unk_49EE2B0;
  if ( result != -8 && result != 0 && result != -16 )
    return sub_1649B30(v39);
  return result;
}
