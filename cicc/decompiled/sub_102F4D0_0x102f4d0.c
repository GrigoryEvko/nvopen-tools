// Function: sub_102F4D0
// Address: 0x102f4d0
//
__int64 __fastcall sub_102F4D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v5; // rax
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // r15
  _BYTE *v9; // r14
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned int v12; // esi
  __int64 v13; // r13
  _QWORD *v14; // rdi
  int v15; // edx
  __int64 v16; // rax
  unsigned __int8 *v17; // rax
  __int64 v18; // r10
  unsigned int v19; // edx
  _QWORD *v20; // rax
  __int64 v21; // r9
  unsigned int v22; // esi
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // r11d
  unsigned int v26; // edi
  __int64 v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rdi
  int v31; // ecx
  int v32; // ecx
  _QWORD *v33; // rax
  int v34; // edi
  int v35; // edi
  unsigned int v36; // esi
  __int64 v37; // r11
  _QWORD *v38; // rdx
  int v39; // eax
  int v40; // r8d
  unsigned int v41; // r13d
  __int64 v42; // rdi
  int v43; // esi
  unsigned __int8 *v44; // [rsp+0h] [rbp-70h]
  __int64 v45; // [rsp+0h] [rbp-70h]
  __int64 v46; // [rsp+0h] [rbp-70h]
  __int64 v47; // [rsp+0h] [rbp-70h]
  __int64 v48; // [rsp+0h] [rbp-70h]
  __int64 v50; // [rsp+8h] [rbp-68h]
  int v51; // [rsp+8h] [rbp-68h]
  _QWORD *v52; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v53[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v54; // [rsp+30h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
    return 0x6000000000000003LL;
  if ( !sub_B91C10(a2, 16) )
    return 0x6000000000000003LL;
  v5 = sub_BD3990(*(unsigned __int8 **)(a2 - 32), 16);
  v44 = v5;
  if ( *v5 <= 3u )
    return 0x6000000000000003LL;
  v7 = *((_QWORD *)v5 + 2);
  if ( !v7 )
    return 0x6000000000000003LL;
  v8 = 0;
  do
  {
    v9 = *(_BYTE **)(v7 + 24);
    if ( *v9 > 0x1Cu && (_BYTE *)a2 != v9 )
    {
      if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 280), *(_QWORD *)(v7 + 24), a2) )
      {
        if ( (*v9 == 61 || *v9 == 62 && (v17 = (unsigned __int8 *)*((_QWORD *)v9 - 4)) != 0 && v44 == v17)
          && (v9[7] & 0x20) != 0
          && sub_B91C10((__int64)v9, 16) )
        {
          if ( v8 )
          {
            if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 280), v8, (__int64)v9) )
              v8 = (__int64)v9;
          }
          else
          {
            v8 = (__int64)v9;
          }
        }
      }
    }
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v7 );
  if ( !v8 )
    return 0x6000000000000003LL;
  v10 = *(_QWORD *)(v8 + 40);
  result = v8 | 2;
  if ( a3 != v10 )
  {
    v53[0] = 0;
    v11 = a1 + 32;
    v53[1] = 0;
    v54 = a2;
    if ( a2 != -4096 && a2 != -8192 )
    {
      v45 = v10;
      sub_BD73F0((__int64)v53);
      v10 = v45;
      v11 = a1 + 32;
    }
    v12 = *(_DWORD *)(a1 + 56);
    if ( v12 )
    {
      v13 = v54;
      v18 = *(_QWORD *)(a1 + 40);
      v19 = (v12 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v20 = (_QWORD *)(v18 + 48LL * v19);
      v21 = v20[2];
      if ( v54 == v21 )
        goto LABEL_41;
      v51 = 1;
      v14 = 0;
      v46 = v10;
      while ( v21 != -4096 )
      {
        if ( v21 == -8192 && !v14 )
          v14 = v20;
        v19 = (v12 - 1) & (v51 + v19);
        v20 = (_QWORD *)(v18 + 48LL * v19);
        v21 = v20[2];
        if ( v54 == v21 )
          goto LABEL_41;
        ++v51;
      }
      if ( !v14 )
        v14 = v20;
      v39 = *(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 32);
      v15 = v39 + 1;
      v52 = v14;
      if ( 4 * (v39 + 1) < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(a1 + 52) - v15 > v12 >> 3 )
        {
LABEL_23:
          *(_DWORD *)(a1 + 48) = v15;
          if ( v14[2] == -4096 )
          {
            if ( v13 != -4096 )
            {
LABEL_28:
              v14[2] = v13;
              if ( v13 == 0 || v13 == -4096 || v13 == -8192 )
              {
                v13 = v54;
              }
              else
              {
                v48 = v10;
                sub_BD73F0((__int64)v14);
                v13 = v54;
                v10 = v48;
              }
            }
          }
          else
          {
            --*(_DWORD *)(a1 + 52);
            v16 = v14[2];
            if ( v16 != v13 )
            {
              if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
              {
                v47 = v10;
                sub_BD60C0(v14);
                v10 = v47;
              }
              goto LABEL_28;
            }
          }
          v14[3] = v10;
          v14[4] = v8 | 2;
          v14[5] = 0;
LABEL_41:
          if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
            sub_BD60C0(v53);
          v22 = *(_DWORD *)(a1 + 88);
          v23 = a1 + 64;
          if ( v22 )
          {
            v24 = *(_QWORD *)(a1 + 72);
            v25 = 1;
            v26 = (v22 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v27 = v24 + 72LL * v26;
            v28 = 0;
            v29 = *(_QWORD *)v27;
            if ( *(_QWORD *)v27 == v8 )
            {
LABEL_46:
              v30 = v27 + 8;
              if ( !*(_BYTE *)(v27 + 36) )
                goto LABEL_47;
              goto LABEL_62;
            }
            while ( v29 != -4096 )
            {
              if ( !v28 && v29 == -8192 )
                v28 = (_QWORD *)v27;
              v26 = (v22 - 1) & (v25 + v26);
              v27 = v24 + 72LL * v26;
              v29 = *(_QWORD *)v27;
              if ( *(_QWORD *)v27 == v8 )
                goto LABEL_46;
              ++v25;
            }
            v31 = *(_DWORD *)(a1 + 80);
            if ( !v28 )
              v28 = (_QWORD *)v27;
            ++*(_QWORD *)(a1 + 64);
            v32 = v31 + 1;
            if ( 4 * v32 < 3 * v22 )
            {
              if ( v22 - *(_DWORD *)(a1 + 84) - v32 > v22 >> 3 )
              {
LABEL_59:
                *(_DWORD *)(a1 + 80) = v32;
                if ( *v28 != -4096 )
                  --*(_DWORD *)(a1 + 84);
                *v28 = v8;
                v30 = (__int64)(v28 + 1);
                v28[1] = 0;
                v28[2] = v28 + 5;
                v28[3] = 4;
                *((_DWORD *)v28 + 8) = 0;
                *((_BYTE *)v28 + 36) = 1;
LABEL_62:
                v33 = *(_QWORD **)(v30 + 8);
                v29 = *(unsigned int *)(v30 + 20);
                v27 = (__int64)&v33[v29];
                if ( v33 != (_QWORD *)v27 )
                {
                  while ( a2 != *v33 )
                  {
                    if ( (_QWORD *)v27 == ++v33 )
                      goto LABEL_65;
                  }
                  return 0x2000000000000003LL;
                }
LABEL_65:
                if ( (unsigned int)v29 < *(_DWORD *)(v30 + 16) )
                {
                  *(_DWORD *)(v30 + 20) = v29 + 1;
                  *(_QWORD *)v27 = a2;
                  ++*(_QWORD *)v30;
                  return 0x2000000000000003LL;
                }
LABEL_47:
                sub_C8CC70(v30, a2, v27, v29, v23, v24);
                return 0x2000000000000003LL;
              }
              sub_102F2A0(a1 + 64, v22);
              v40 = *(_DWORD *)(a1 + 88);
              if ( v40 )
              {
                v23 = (unsigned int)(v40 - 1);
                v24 = *(_QWORD *)(a1 + 72);
                v41 = v23 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
                v32 = *(_DWORD *)(a1 + 80) + 1;
                v28 = (_QWORD *)(v24 + 72LL * v41);
                v42 = *v28;
                if ( *v28 == v8 )
                  goto LABEL_59;
                v38 = (_QWORD *)(v24 + 72LL * v41);
                v43 = 1;
                v28 = 0;
                while ( v42 != -4096 )
                {
                  if ( !v28 && v42 == -8192 )
                    v28 = v38;
                  v41 = v23 & (v43 + v41);
                  v38 = (_QWORD *)(v24 + 72LL * v41);
                  v42 = *v38;
                  if ( *v38 == v8 )
                    goto LABEL_94;
                  ++v43;
                }
                goto LABEL_76;
              }
              goto LABEL_106;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 64);
          }
          sub_102F2A0(a1 + 64, 2 * v22);
          v34 = *(_DWORD *)(a1 + 88);
          if ( v34 )
          {
            v35 = v34 - 1;
            v24 = *(_QWORD *)(a1 + 72);
            v36 = v35 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v32 = *(_DWORD *)(a1 + 80) + 1;
            v28 = (_QWORD *)(v24 + 72LL * v36);
            v37 = *v28;
            if ( *v28 == v8 )
              goto LABEL_59;
            v38 = (_QWORD *)(v24 + 72LL * (v35 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))));
            v23 = 1;
            v28 = 0;
            while ( v37 != -4096 )
            {
              if ( !v28 && v37 == -8192 )
                v28 = v38;
              v36 = v35 & (v23 + v36);
              v38 = (_QWORD *)(v24 + 72LL * v36);
              v37 = *v38;
              if ( *v38 == v8 )
              {
LABEL_94:
                v28 = v38;
                goto LABEL_59;
              }
              v23 = (unsigned int)(v23 + 1);
            }
LABEL_76:
            if ( !v28 )
              v28 = v38;
            goto LABEL_59;
          }
LABEL_106:
          ++*(_DWORD *)(a1 + 80);
          BUG();
        }
LABEL_22:
        v50 = v11;
        sub_102EF10(v11, v12);
        sub_102D260(v50, (__int64)v53, &v52);
        v13 = v54;
        v14 = v52;
        v10 = v46;
        v15 = *(_DWORD *)(a1 + 48) + 1;
        goto LABEL_23;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
      v52 = 0;
    }
    v46 = v10;
    v12 *= 2;
    goto LABEL_22;
  }
  return result;
}
