// Function: sub_D1E880
// Address: 0xd1e880
//
__int64 __fastcall sub_D1E880(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 v9; // r12
  _BYTE *v10; // rsi
  char v11; // al
  unsigned int v12; // r12d
  __int64 *v14; // rax
  __int64 v15; // r15
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  _QWORD *v18; // rdi
  __int64 v19; // rax
  unsigned __int8 *v20; // rdi
  __int64 v21; // r9
  _BYTE *v22; // rsi
  _BYTE *v23; // r12
  __int64 v24; // rcx
  __int64 v25; // rdi
  _QWORD *v26; // r9
  int v27; // r14d
  unsigned int v28; // edx
  _QWORD *v29; // rax
  __int64 v30; // r11
  __int64 *v31; // rax
  _BYTE *v32; // r14
  __int64 v33; // r15
  _QWORD *v34; // r12
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned int v37; // esi
  int v38; // eax
  __int64 v39; // rsi
  int v40; // r8d
  __int64 v41; // rdi
  int v42; // edx
  unsigned int v43; // ecx
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // eax
  int v47; // eax
  __int64 v48; // rdi
  int v49; // esi
  _QWORD *v50; // r10
  __int64 v51; // r8
  int v52; // r11d
  unsigned int v53; // ecx
  __int64 v54; // rax
  int v55; // r11d
  __int64 v56; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v57; // [rsp+18h] [rbp-58h] BYREF
  _BYTE *v58; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v59; // [rsp+28h] [rbp-48h]
  _BYTE *v60; // [rsp+30h] [rbp-40h]

  v8 = *(_QWORD *)(a2 - 32);
  v58 = 0;
  v59 = 0;
  v60 = 0;
  if ( v8 )
  {
    if ( sub_AC30F0(v8) )
    {
      v9 = *(_QWORD *)(a2 + 16);
      if ( v9 )
        goto LABEL_4;
LABEL_31:
      v23 = v59;
      if ( v58 != v59 )
      {
        v56 = a1 + 240;
        while ( 1 )
        {
          v37 = *(_DWORD *)(a1 + 264);
          if ( !v37 )
            break;
          v24 = *((_QWORD *)v23 - 1);
          v25 = *(_QWORD *)(a1 + 248);
          v26 = 0;
          v27 = 1;
          v28 = (v37 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v29 = (_QWORD *)(v25 + 16LL * v28);
          v30 = *v29;
          if ( v24 != *v29 )
          {
            while ( v30 != -4096 )
            {
              if ( !v26 && v30 == -8192 )
                v26 = v29;
              v28 = (v37 - 1) & (v27 + v28);
              v29 = (_QWORD *)(v25 + 16LL * v28);
              v30 = *v29;
              if ( v24 == *v29 )
                goto LABEL_34;
              ++v27;
            }
            if ( !v26 )
              v26 = v29;
            v46 = *(_DWORD *)(a1 + 256);
            ++*(_QWORD *)(a1 + 240);
            v42 = v46 + 1;
            if ( 4 * (v46 + 1) < 3 * v37 )
            {
              if ( v37 - *(_DWORD *)(a1 + 260) - v42 <= v37 >> 3 )
              {
                sub_D1E6A0(v56, v37);
                v47 = *(_DWORD *)(a1 + 264);
                if ( !v47 )
                {
LABEL_80:
                  ++*(_DWORD *)(a1 + 256);
                  BUG();
                }
                v48 = *((_QWORD *)v23 - 1);
                v49 = v47 - 1;
                v50 = 0;
                v51 = *(_QWORD *)(a1 + 248);
                v52 = 1;
                v42 = *(_DWORD *)(a1 + 256) + 1;
                v53 = (v47 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                v26 = (_QWORD *)(v51 + 16LL * v53);
                v54 = *v26;
                if ( v48 != *v26 )
                {
                  while ( v54 != -4096 )
                  {
                    if ( !v50 && v54 == -8192 )
                      v50 = v26;
                    v53 = v49 & (v52 + v53);
                    v26 = (_QWORD *)(v51 + 16LL * v53);
                    v54 = *v26;
                    if ( v48 == *v26 )
                      goto LABEL_43;
                    ++v52;
                  }
LABEL_61:
                  if ( v50 )
                    v26 = v50;
                }
              }
LABEL_43:
              *(_DWORD *)(a1 + 256) = v42;
              if ( *v26 != -4096 )
                --*(_DWORD *)(a1 + 260);
              v45 = *((_QWORD *)v23 - 1);
              v26[1] = 0;
              *v26 = v45;
              v31 = v26 + 1;
              goto LABEL_35;
            }
LABEL_41:
            sub_D1E6A0(v56, 2 * v37);
            v38 = *(_DWORD *)(a1 + 264);
            if ( !v38 )
              goto LABEL_80;
            v39 = *((_QWORD *)v23 - 1);
            v40 = v38 - 1;
            v41 = *(_QWORD *)(a1 + 248);
            v42 = *(_DWORD *)(a1 + 256) + 1;
            v43 = (v38 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
            v26 = (_QWORD *)(v41 + 16LL * v43);
            v44 = *v26;
            if ( v39 != *v26 )
            {
              v55 = 1;
              v50 = 0;
              while ( v44 != -4096 )
              {
                if ( v44 == -8192 && !v50 )
                  v50 = v26;
                v43 = v40 & (v55 + v43);
                v26 = (_QWORD *)(v41 + 16LL * v43);
                v44 = *v26;
                if ( v39 == *v26 )
                  goto LABEL_43;
                ++v55;
              }
              goto LABEL_61;
            }
            goto LABEL_43;
          }
LABEL_34:
          v31 = v29 + 1;
LABEL_35:
          *v31 = a2;
          v32 = v59;
          v33 = *(_QWORD *)(a1 + 336);
          v34 = (_QWORD *)sub_22077B0(64);
          v35 = *((_QWORD *)v32 - 1);
          v34[3] = 2;
          v34[5] = v35;
          v34[4] = 0;
          if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
            sub_BD73F0((__int64)(v34 + 3));
          v34[6] = a1;
          v34[7] = 0;
          v34[2] = &unk_49DDE50;
          sub_2208C80(v34, v33);
          v36 = *(_QWORD *)(a1 + 336);
          ++*(_QWORD *)(a1 + 352);
          *(_QWORD *)(v36 + 56) = v36;
          v23 = v59 - 8;
          v59 = v23;
          if ( v58 == v23 )
            goto LABEL_12;
        }
        ++*(_QWORD *)(a1 + 240);
        goto LABEL_41;
      }
      goto LABEL_12;
    }
LABEL_7:
    v12 = 0;
    goto LABEL_8;
  }
  v9 = *(_QWORD *)(a2 + 16);
  if ( v9 )
  {
LABEL_4:
    while ( 1 )
    {
      v10 = *(_BYTE **)(v9 + 24);
      v11 = *v10;
      if ( *v10 <= 0x1Cu )
        goto LABEL_7;
      if ( v11 == 61 )
      {
        if ( (unsigned __int8)sub_D1B9A0(a1, (__int64)v10, 0, 0, 0, a6) )
          goto LABEL_7;
      }
      else
      {
        if ( v11 != 62 )
          goto LABEL_7;
        v20 = (unsigned __int8 *)*((_QWORD *)v10 - 8);
        if ( !v20 )
          BUG();
        if ( v20 == (unsigned __int8 *)a2 )
          goto LABEL_7;
        if ( *v20 != 20 )
        {
          v57 = sub_98ACB0(v20, 6u);
          if ( !(unsigned __int8)sub_CF6FD0(v57) || (unsigned __int8)sub_D1B9A0(a1, (__int64)v57, 0, 0, a2, v21) )
            goto LABEL_7;
          v22 = v59;
          if ( v59 == v60 )
          {
            sub_9281F0((__int64)&v58, v59, &v57);
          }
          else
          {
            if ( v59 )
            {
              *(_QWORD *)v59 = v57;
              v22 = v59;
            }
            v59 = v22 + 8;
          }
        }
      }
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        goto LABEL_31;
    }
  }
LABEL_12:
  if ( !*(_BYTE *)(a1 + 172) )
    goto LABEL_65;
  v14 = *(__int64 **)(a1 + 152);
  a4 = *(unsigned int *)(a1 + 164);
  a3 = &v14[a4];
  if ( v14 != a3 )
  {
    while ( *v14 != a2 )
    {
      if ( a3 == ++v14 )
        goto LABEL_66;
    }
    goto LABEL_17;
  }
LABEL_66:
  if ( (unsigned int)a4 < *(_DWORD *)(a1 + 160) )
  {
    *(_DWORD *)(a1 + 164) = a4 + 1;
    *a3 = a2;
    ++*(_QWORD *)(a1 + 144);
  }
  else
  {
LABEL_65:
    sub_C8CC70(a1 + 144, a2, (__int64)a3, a4, a5, a6);
  }
LABEL_17:
  v15 = *(_QWORD *)(a1 + 336);
  v16 = (_QWORD *)sub_22077B0(64);
  v16[3] = 2;
  v17 = v16;
  v16[4] = 0;
  v16[5] = a2;
  if ( a2 != -8192 && a2 != -4096 )
    sub_BD73F0((__int64)(v16 + 3));
  v17[6] = a1;
  v18 = v17;
  v17[2] = &unk_49DDE50;
  v17[7] = 0;
  v12 = 1;
  sub_2208C80(v18, v15);
  v19 = *(_QWORD *)(a1 + 336);
  ++*(_QWORD *)(a1 + 352);
  *(_QWORD *)(v19 + 56) = v19;
LABEL_8:
  if ( v58 )
    j_j___libc_free_0(v58, v60 - v58);
  return v12;
}
