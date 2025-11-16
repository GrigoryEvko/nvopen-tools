// Function: sub_1878AE0
// Address: 0x1878ae0
//
__int64 *__fastcall sub_1878AE0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r10d
  __int64 *v8; // r9
  unsigned int v9; // edx
  __int64 v10; // rcx
  int v12; // ecx
  int v13; // ecx
  __int64 v14; // r15
  unsigned __int64 v15; // r15
  _QWORD *v16; // r8
  _QWORD *v17; // r12
  __int64 v18; // r9
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // r15
  _BYTE *v22; // r14
  __int64 v23; // rdx
  __int64 *v24; // rax
  unsigned __int64 *v25; // r15
  unsigned __int64 v26; // rdx
  _QWORD *v27; // r8
  _QWORD *v28; // r13
  _QWORD *v29; // r12
  unsigned __int64 v30; // rsi
  _QWORD *v31; // rax
  _QWORD *v32; // rsi
  unsigned __int64 v33; // rdx
  __int64 v34; // rcx
  _QWORD *v35; // r8
  _BOOL4 v36; // r11d
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  _BOOL4 v39; // r14d
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  _BYTE *v42; // rdi
  _BYTE *v43; // rax
  _QWORD *v44; // rdx
  _QWORD *v45; // r11
  _BYTE *v46; // rdi
  _BYTE *v47; // rax
  _QWORD *v48; // r11
  __int64 *v49; // [rsp+8h] [rbp-A8h]
  _BOOL4 v50; // [rsp+14h] [rbp-9Ch]
  _QWORD *v51; // [rsp+18h] [rbp-98h]
  _QWORD *v52; // [rsp+18h] [rbp-98h]
  unsigned __int64 v53; // [rsp+20h] [rbp-90h]
  __int64 v54; // [rsp+20h] [rbp-90h]
  __int64 v55; // [rsp+20h] [rbp-90h]
  unsigned __int64 *v56; // [rsp+28h] [rbp-88h]
  _QWORD *v57; // [rsp+28h] [rbp-88h]
  _QWORD *v58; // [rsp+28h] [rbp-88h]
  __int64 v59; // [rsp+38h] [rbp-78h] BYREF
  __int64 *v60; // [rsp+48h] [rbp-68h] BYREF
  __int64 v61; // [rsp+50h] [rbp-60h] BYREF
  __int64 v62; // [rsp+58h] [rbp-58h]
  unsigned __int64 v63; // [rsp+60h] [rbp-50h]
  __int64 v64; // [rsp+68h] [rbp-48h]
  char v65; // [rsp+70h] [rbp-40h]

  v2 = a2;
  v4 = *a1;
  v59 = a2;
  v61 = a2;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v5 = *(_DWORD *)(v4 + 136);
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 112);
LABEL_75:
    sub_1874910(v4 + 112, 2 * v5);
    goto LABEL_76;
  }
  v6 = *(_QWORD *)(v4 + 120);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v49 = (__int64 *)(v6 + 40LL * v9);
  v10 = *v49;
  if ( v2 == *v49 )
    return v49 + 1;
  while ( v10 != -4 )
  {
    if ( v10 == -8 && !v8 )
      v8 = v49;
    v9 = (v5 - 1) & (v7 + v9);
    v49 = (__int64 *)(v6 + 40LL * v9);
    v10 = *v49;
    if ( v2 == *v49 )
      return v49 + 1;
    ++v7;
  }
  v12 = *(_DWORD *)(v4 + 128);
  if ( !v8 )
    v8 = v49;
  ++*(_QWORD *)(v4 + 112);
  v13 = v12 + 1;
  v49 = v8;
  if ( 4 * v13 >= 3 * v5 )
    goto LABEL_75;
  if ( v5 - *(_DWORD *)(v4 + 132) - v13 > v5 >> 3 )
    goto LABEL_14;
  sub_1874910(v4 + 112, v5);
LABEL_76:
  sub_1872280(v4 + 112, &v61, &v60);
  v49 = v60;
  v13 = *(_DWORD *)(v4 + 128) + 1;
  v2 = v61;
LABEL_14:
  *(_DWORD *)(v4 + 128) = v13;
  if ( *v49 != -4 )
    --*(_DWORD *)(v4 + 132);
  *v49 = v2;
  v14 = v59;
  v49[1] = v62;
  v15 = v14 | 2;
  v49[2] = v63;
  v49[3] = v64;
  *((_BYTE *)v49 + 32) = v65;
  v16 = (_QWORD *)a1[1];
  v62 = 1;
  v61 = (__int64)&v61;
  v63 = v15;
  v17 = (_QWORD *)v16[2];
  v18 = (__int64)(v16 + 1);
  if ( !v17 )
  {
    v17 = v16 + 1;
    if ( v18 == v16[3] )
    {
      v39 = 1;
LABEL_64:
      v57 = v16;
      v54 = v18;
      v21 = (_QWORD *)sub_22077B0(56);
      v21[4] = v21 + 4;
      v40 = v63;
      v21[5] = 1;
      v21[6] = v40;
      sub_220F040(v39, v21, v17, v54);
      ++v57[5];
      v18 = a1[1] + 8;
      goto LABEL_25;
    }
LABEL_67:
    v55 = (__int64)(v16 + 1);
    v58 = v16;
    v41 = sub_220EF80(v17);
    v16 = v58;
    v18 = v55;
    if ( *(_QWORD *)(v41 + 48) >= v15 )
    {
      v17 = (_QWORD *)v41;
      goto LABEL_24;
    }
LABEL_62:
    v39 = 1;
    if ( (_QWORD *)v18 != v17 )
      v39 = v15 < v17[6];
    goto LABEL_64;
  }
  while ( 1 )
  {
    v19 = v17[6];
    v20 = (_QWORD *)v17[3];
    if ( v19 > v15 )
      v20 = (_QWORD *)v17[2];
    if ( !v20 )
      break;
    v17 = v20;
  }
  if ( v15 < v19 )
  {
    if ( v17 == (_QWORD *)v16[3] )
      goto LABEL_62;
    goto LABEL_67;
  }
  if ( v19 < v15 )
    goto LABEL_62;
LABEL_24:
  v21 = v17;
LABEL_25:
  v22 = 0;
  if ( v21 != (_QWORD *)v18 )
  {
    v22 = v21 + 4;
    if ( (v21[5] & 1) == 0 )
    {
      v22 = (_BYTE *)v21[4];
      if ( (v22[8] & 1) == 0 )
      {
        v23 = *(_QWORD *)v22;
        if ( (*(_BYTE *)(*(_QWORD *)v22 + 8LL) & 1) != 0 )
        {
          v22 = *(_BYTE **)v22;
        }
        else
        {
          v42 = *(_BYTE **)v23;
          if ( (*(_BYTE *)(*(_QWORD *)v23 + 8LL) & 1) == 0 )
          {
            v43 = sub_1874270(v42);
            *v44 = v43;
            v42 = v43;
          }
          *(_QWORD *)v22 = v42;
          v22 = v42;
        }
        v21[4] = v22;
      }
    }
  }
  v24 = sub_1874D50(a1[2], &v59);
  v25 = (unsigned __int64 *)v24[2];
  v56 = (unsigned __int64 *)v24[3];
  if ( v56 != v25 )
  {
    while ( 1 )
    {
      v26 = *v25;
      v27 = (_QWORD *)a1[1];
      v62 = 1;
      v63 = v26;
      v28 = v27 + 1;
      v61 = (__int64)&v61;
      v29 = (_QWORD *)v27[2];
      if ( v29 )
        break;
      v29 = v27 + 1;
      if ( v28 != (_QWORD *)v27[3] )
        goto LABEL_56;
      v36 = 1;
LABEL_54:
      v51 = v27;
      v50 = v36;
      v32 = (_QWORD *)sub_22077B0(56);
      v32[4] = v32 + 4;
      v37 = v63;
      v32[5] = 1;
      v32[6] = v37;
      sub_220F040(v50, v32, v29, v28);
      ++v51[5];
LABEL_41:
      v33 = 0;
      if ( v32 != v28 )
        goto LABEL_42;
LABEL_48:
      if ( v22 != (_BYTE *)v33 )
      {
        *(_QWORD *)(*(_QWORD *)v22 + 8LL) = v33 | *(_QWORD *)(*(_QWORD *)v22 + 8LL) & 1LL;
        *(_QWORD *)v22 = *(_QWORD *)v33;
        *(_QWORD *)(v33 + 8) &= ~1uLL;
        *(_QWORD *)v33 = v22;
      }
      if ( v56 == ++v25 )
        return v49 + 1;
    }
    while ( 1 )
    {
      v30 = v29[6];
      v31 = (_QWORD *)v29[3];
      if ( v26 < v30 )
        v31 = (_QWORD *)v29[2];
      if ( !v31 )
        break;
      v29 = v31;
    }
    if ( v26 < v30 )
    {
      if ( (_QWORD *)v27[3] != v29 )
      {
LABEL_56:
        v53 = v26;
        v52 = v27;
        v38 = sub_220EF80(v29);
        v26 = v53;
        if ( v53 <= *(_QWORD *)(v38 + 48) )
        {
          v29 = (_QWORD *)v38;
LABEL_40:
          v32 = v29;
          goto LABEL_41;
        }
        v27 = v52;
        if ( !v29 )
        {
          v32 = 0;
LABEL_42:
          v33 = (unsigned __int64)(v32 + 4);
          if ( (v32[5] & 1) == 0 )
          {
            v33 = v32[4];
            if ( (*(_BYTE *)(v33 + 8) & 1) == 0 )
            {
              v34 = *(_QWORD *)v33;
              if ( (*(_BYTE *)(*(_QWORD *)v33 + 8LL) & 1) != 0 )
              {
                v33 = *(_QWORD *)v33;
              }
              else
              {
                v35 = *(_QWORD **)v34;
                if ( (*(_BYTE *)(*(_QWORD *)v34 + 8LL) & 1) == 0 )
                {
                  v45 = (_QWORD *)*v35;
                  if ( (*(_BYTE *)(*v35 + 8LL) & 1) != 0 )
                  {
                    v35 = (_QWORD *)*v35;
                  }
                  else
                  {
                    v46 = (_BYTE *)*v45;
                    if ( (*(_BYTE *)(*v45 + 8LL) & 1) == 0 )
                    {
                      v47 = sub_1874270(v46);
                      *v48 = v47;
                      v46 = v47;
                    }
                    *v35 = v46;
                    v35 = v46;
                  }
                  *(_QWORD *)v34 = v35;
                }
                *(_QWORD *)v33 = v35;
                v33 = (unsigned __int64)v35;
              }
              v32[4] = v33;
            }
          }
          goto LABEL_48;
        }
      }
    }
    else if ( v26 <= v30 )
    {
      goto LABEL_40;
    }
    v36 = 1;
    if ( v29 != v28 )
      v36 = v26 < v29[6];
    goto LABEL_54;
  }
  return v49 + 1;
}
