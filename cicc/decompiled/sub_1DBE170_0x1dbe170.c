// Function: sub_1DBE170
// Address: 0x1dbe170
//
unsigned __int64 __fastcall sub_1DBE170(__int64 a1, unsigned __int8 *a2, int a3)
{
  unsigned __int64 result; // rax
  unsigned __int8 *v5; // r12
  int v6; // r15d
  __int64 v7; // r14
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // r13
  __int64 *v13; // r10
  unsigned __int64 *v14; // rdx
  char v15; // dl
  __int64 v16; // rcx
  unsigned int v17; // edx
  __int64 v18; // rcx
  unsigned __int16 *v19; // rdx
  int v20; // r15d
  unsigned __int16 *v21; // r14
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 *v24; // rax
  char v25; // dl
  char v26; // al
  _QWORD *v27; // rax
  _QWORD *v28; // r9
  __int64 v29; // rdx
  __int64 *v30; // rsi
  unsigned int v31; // edi
  __int64 *v32; // rcx
  unsigned int v33; // eax
  __int64 v34; // rcx
  __int64 *v35; // rsi
  unsigned int v36; // edi
  __int64 *v37; // rcx
  unsigned __int16 v38; // ax
  int v39; // r14d
  int v40; // r12d
  char v41; // dl
  char v42; // r11
  int v43; // r14d
  __int64 *v44; // rdi
  unsigned int v45; // r11d
  __int64 *v46; // rax
  __int64 *v47; // rsi
  int v48; // eax
  __int64 v49; // rdx
  unsigned __int64 *v50; // r8
  __int64 v51; // rcx
  __int64 *v52; // rsi
  __int64 v53; // r8
  __int64 v54; // rdi
  _QWORD *v55; // rsi
  _QWORD *v56; // rdx
  __int64 v57; // rax
  _QWORD *v58; // [rsp+8h] [rbp-58h]
  __int64 v59; // [rsp+8h] [rbp-58h]
  __int64 v60; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v61; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v62; // [rsp+10h] [rbp-50h]
  __int64 v63; // [rsp+10h] [rbp-50h]
  unsigned int v64; // [rsp+10h] [rbp-50h]
  char v65; // [rsp+1Fh] [rbp-41h]
  unsigned __int8 *v66; // [rsp+20h] [rbp-40h]
  __int64 v67; // [rsp+28h] [rbp-38h]
  char v68; // [rsp+28h] [rbp-38h]
  __int64 v69; // [rsp+28h] [rbp-38h]
  __int64 v70; // [rsp+28h] [rbp-38h]
  __int64 v71; // [rsp+28h] [rbp-38h]

  result = (unsigned __int64)&a2[40 * a3];
  v66 = (unsigned __int8 *)result;
  v65 = 0;
  if ( (unsigned __int8 *)result == a2 )
    return result;
  v5 = a2;
  while ( 2 )
  {
    while ( 2 )
    {
      while ( 2 )
      {
        while ( 1 )
        {
          result = *v5;
          if ( (_BYTE)result != 12 )
            break;
          v65 = 1;
          v5 += 40;
          if ( v66 == v5 )
            goto LABEL_20;
        }
        if ( (_BYTE)result )
          goto LABEL_19;
        result = v5[3];
        if ( (result & 0x10) == 0 )
        {
          if ( (v5[4] & 1) != 0 || (v5[4] & 2) != 0 )
            goto LABEL_19;
          result = (unsigned int)result & 0xFFFFFFBF;
          v5[3] = result;
        }
        v6 = *((_DWORD *)v5 + 2);
        if ( !v6 )
          goto LABEL_19;
        if ( v6 >= 0 )
        {
          v16 = *(_QWORD *)(a1 + 16);
          v61 = v5;
          v17 = *(_DWORD *)(*(_QWORD *)(v16 + 8) + 24LL * (unsigned int)v6 + 16);
          result = v17 & 0xF;
          v20 = result * v6;
          v18 = *(_QWORD *)(v16 + 56) + 2LL * (v17 >> 4);
          v19 = (unsigned __int16 *)(v18 + 2);
          LOWORD(v20) = *(_WORD *)v18 + v20;
LABEL_25:
          v21 = v19;
          if ( !v19 )
          {
LABEL_35:
            v5 += 40;
            if ( v66 == v61 + 40 )
              goto LABEL_20;
            continue;
          }
          while ( 1 )
          {
            v22 = (unsigned __int16)v20;
            if ( !*(_BYTE *)(a1 + 144) )
              break;
            v26 = sub_1E6A680(*(_QWORD *)(a1 + 8), (unsigned __int16)v20, (unsigned __int16)v20, v18);
            v22 = (unsigned __int16)v20;
            if ( v26 )
              break;
            v23 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 672LL) + 8LL * (unsigned __int16)v20);
            if ( !v23 )
            {
              v58 = *(_QWORD **)a1;
              v68 = qword_4FC4440[20];
              v27 = (_QWORD *)sub_22077B0(104);
              v28 = v58;
              v29 = (unsigned __int16)v20;
              v23 = (__int64)v27;
              if ( v27 )
              {
                *v27 = v27 + 2;
                v27[1] = 0x200000000LL;
                v27[8] = v27 + 10;
                v27[9] = 0x200000000LL;
                if ( v68 )
                {
                  v57 = sub_22077B0(48);
                  v28 = v58;
                  v29 = (unsigned __int16)v20;
                  if ( v57 )
                  {
                    *(_DWORD *)(v57 + 8) = 0;
                    *(_QWORD *)(v57 + 16) = 0;
                    *(_QWORD *)(v57 + 24) = v57 + 8;
                    *(_QWORD *)(v57 + 32) = v57 + 8;
                    *(_QWORD *)(v57 + 40) = 0;
                  }
                  *(_QWORD *)(v23 + 96) = v57;
                }
                else
                {
                  v27[12] = 0;
                }
              }
              *(_QWORD *)(v28[84] + 8 * v29) = v23;
              sub_1DBA8F0(v28, v23, (unsigned __int16)v20);
LABEL_28:
              if ( !v23 )
                goto LABEL_33;
            }
            v24 = *(__int64 **)(a1 + 48);
            if ( *(__int64 **)(a1 + 56) == v24 )
            {
              v30 = &v24[*(unsigned int *)(a1 + 68)];
              v31 = *(_DWORD *)(a1 + 68);
              if ( v24 != v30 )
              {
                v32 = 0;
                while ( *v24 != v23 )
                {
                  if ( *v24 == -2 )
                    v32 = v24;
                  if ( v30 == ++v24 )
                  {
                    if ( !v32 )
                      goto LABEL_52;
                    *v32 = v23;
                    --*(_DWORD *)(a1 + 72);
                    ++*(_QWORD *)(a1 + 40);
                    goto LABEL_31;
                  }
                }
                goto LABEL_33;
              }
LABEL_52:
              if ( v31 < *(_DWORD *)(a1 + 64) )
              {
                *(_DWORD *)(a1 + 68) = v31 + 1;
                *v30 = v23;
                ++*(_QWORD *)(a1 + 40);
LABEL_31:
                if ( *(_DWORD *)((*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= *(_DWORD *)((*(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL)
                                                                                                  + 24) )
                  sub_1DBDCF0((_QWORD *)a1, v23, (unsigned __int16)v20, 0);
                else
                  sub_1DBD310(a1, v23);
                goto LABEL_33;
              }
            }
            sub_16CCBA0(a1 + 40, v23);
            if ( v25 )
              goto LABEL_31;
LABEL_33:
            result = *v21;
            v19 = 0;
            ++v21;
            v18 = (unsigned int)(result + v20);
            if ( !(_WORD)result )
              goto LABEL_25;
            v20 += result;
            if ( !v21 )
              goto LABEL_35;
          }
          v23 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 672LL) + 8 * v22);
          goto LABEL_28;
        }
        break;
      }
      v7 = *(_QWORD *)a1;
      v8 = v6 & 0x7FFFFFFF;
      v9 = *(unsigned int *)(*(_QWORD *)a1 + 408LL);
      v10 = v6 & 0x7FFFFFFF;
      if ( (v6 & 0x7FFFFFFFu) < (unsigned int)v9 )
      {
        v11 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8LL * v8);
        if ( v11 )
          goto LABEL_13;
      }
      v33 = v8 + 1;
      if ( (unsigned int)v9 < v33 )
      {
        v53 = v33;
        if ( v33 < v9 )
        {
          *(_DWORD *)(v7 + 408) = v33;
        }
        else if ( v33 > v9 )
        {
          if ( v33 > (unsigned __int64)*(unsigned int *)(v7 + 412) )
          {
            v64 = v33;
            v71 = v33;
            sub_16CD150(v7 + 400, (const void *)(v7 + 416), v33, 8, v33, v10);
            v9 = *(unsigned int *)(v7 + 408);
            v10 = v6 & 0x7FFFFFFF;
            v33 = v64;
            v53 = v71;
          }
          v34 = *(_QWORD *)(v7 + 400);
          v54 = *(_QWORD *)(v7 + 416);
          v55 = (_QWORD *)(v34 + 8 * v53);
          v56 = (_QWORD *)(v34 + 8 * v9);
          if ( v55 != v56 )
          {
            do
              *v56++ = v54;
            while ( v55 != v56 );
            v34 = *(_QWORD *)(v7 + 400);
          }
          *(_DWORD *)(v7 + 408) = v33;
          goto LABEL_56;
        }
      }
      v34 = *(_QWORD *)(v7 + 400);
LABEL_56:
      v69 = v10;
      *(_QWORD *)(v34 + 8LL * (v6 & 0x7FFFFFFF)) = sub_1DBA290(v6);
      v70 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8 * v69);
      sub_1DBB110((_QWORD *)v7, v70);
      v11 = v70;
LABEL_13:
      v12 = *(_QWORD *)(v11 + 104);
      v67 = a1 + 40;
      if ( !v12 )
        goto LABEL_14;
      v38 = (*(_DWORD *)v5 >> 8) & 0xFFF;
      if ( v38 )
      {
        v39 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 248LL) + 4LL * v38);
      }
      else
      {
        v63 = v11;
        v48 = sub_1E69F40(*(_QWORD *)(a1 + 8), (unsigned int)v6);
        v11 = v63;
        v39 = v48;
        v12 = *(_QWORD *)(v63 + 104);
        if ( !v12 )
        {
LABEL_14:
          v13 = *(__int64 **)(a1 + 56);
          v14 = *(unsigned __int64 **)(a1 + 48);
          goto LABEL_15;
        }
      }
      v13 = *(__int64 **)(a1 + 56);
      v14 = *(unsigned __int64 **)(a1 + 48);
      v62 = v5;
      v40 = v39;
      v59 = v11;
      do
      {
        while ( 1 )
        {
          if ( (*(_DWORD *)(v12 + 112) & v40) == 0 )
            goto LABEL_70;
          v43 = *(_DWORD *)(v12 + 112);
          if ( v13 == (__int64 *)v14 )
          {
            v44 = &v13[*(unsigned int *)(a1 + 68)];
            v45 = *(_DWORD *)(a1 + 68);
            if ( v44 != v13 )
            {
              v46 = v13;
              v47 = 0;
              while ( v12 != *v46 )
              {
                if ( *v46 == -2 )
                  v47 = v46;
                if ( v44 == ++v46 )
                {
                  if ( !v47 )
                    goto LABEL_107;
                  *v47 = v12;
                  --*(_DWORD *)(a1 + 72);
                  ++*(_QWORD *)(a1 + 40);
                  goto LABEL_68;
                }
              }
              goto LABEL_70;
            }
LABEL_107:
            if ( v45 < *(_DWORD *)(a1 + 64) )
              break;
          }
          sub_16CCBA0(v67, v12);
          v13 = *(__int64 **)(a1 + 56);
          v42 = v41;
          v14 = *(unsigned __int64 **)(a1 + 48);
          if ( v42 )
            goto LABEL_68;
LABEL_70:
          v12 = *(_QWORD *)(v12 + 104);
          if ( !v12 )
            goto LABEL_86;
        }
        *(_DWORD *)(a1 + 68) = v45 + 1;
        *v44 = v12;
        ++*(_QWORD *)(a1 + 40);
LABEL_68:
        if ( *(_DWORD *)((*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((*(_QWORD *)(a1 + 32)
                                                                                          & 0xFFFFFFFFFFFFFFF8LL)
                                                                                         + 24) )
        {
          sub_1DBD310(a1, v12);
          v13 = *(__int64 **)(a1 + 56);
          v14 = *(unsigned __int64 **)(a1 + 48);
          goto LABEL_70;
        }
        sub_1DBDCF0((_QWORD *)a1, v12, v6, v43);
        v12 = *(_QWORD *)(v12 + 104);
        v13 = *(__int64 **)(a1 + 56);
        v14 = *(unsigned __int64 **)(a1 + 48);
      }
      while ( v12 );
LABEL_86:
      v5 = v62;
      v11 = v59;
LABEL_15:
      if ( v13 != (__int64 *)v14 )
      {
LABEL_16:
        v60 = v11;
        result = (unsigned __int64)sub_16CCBA0(v67, v11);
        v11 = v60;
        if ( v15 )
          goto LABEL_17;
LABEL_19:
        v5 += 40;
        if ( v66 == v5 )
          goto LABEL_20;
        continue;
      }
      break;
    }
    v35 = (__int64 *)&v14[*(unsigned int *)(a1 + 68)];
    v36 = *(_DWORD *)(a1 + 68);
    if ( v35 != (__int64 *)v14 )
    {
      v37 = 0;
      while ( 1 )
      {
        result = *v14;
        if ( v11 == *v14 )
          goto LABEL_19;
        if ( result == -2 )
          v37 = (__int64 *)v14;
        if ( v35 == (__int64 *)++v14 )
        {
          if ( !v37 )
            break;
          *v37 = v11;
          --*(_DWORD *)(a1 + 72);
          ++*(_QWORD *)(a1 + 40);
          goto LABEL_17;
        }
      }
    }
    if ( v36 >= *(_DWORD *)(a1 + 64) )
      goto LABEL_16;
    *(_DWORD *)(a1 + 68) = v36 + 1;
    *v35 = v11;
    ++*(_QWORD *)(a1 + 40);
LABEL_17:
    if ( *(_DWORD *)((*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((*(_QWORD *)(a1 + 32)
                                                                                      & 0xFFFFFFFFFFFFFFF8LL)
                                                                                     + 24) )
    {
      result = sub_1DBD310(a1, v11);
      goto LABEL_19;
    }
    result = sub_1DBDCF0((_QWORD *)a1, v11, v6, 0);
    v5 += 40;
    if ( v66 != v5 )
      continue;
    break;
  }
LABEL_20:
  if ( v65 )
  {
    v49 = *(unsigned int *)(*(_QWORD *)a1 + 440LL);
    v50 = *(unsigned __int64 **)(*(_QWORD *)a1 + 432LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 440LL) )
    {
      do
      {
        while ( 1 )
        {
          v51 = v49 >> 1;
          v52 = (__int64 *)&v50[v49 >> 1];
          if ( (*(_DWORD *)((*v52 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v52 >> 1) & 3) >= (*(_DWORD *)((*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*(__int64 *)(a1 + 24) >> 1) & 3)) )
            break;
          v50 = (unsigned __int64 *)(v52 + 1);
          v49 = v49 - v51 - 1;
          if ( v49 <= 0 )
            goto LABEL_92;
        }
        v49 >>= 1;
      }
      while ( v51 > 0 );
    }
LABEL_92:
    result = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL | 4;
    *v50 = result;
  }
  return result;
}
