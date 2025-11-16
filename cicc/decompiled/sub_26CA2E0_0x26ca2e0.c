// Function: sub_26CA2E0
// Address: 0x26ca2e0
//
__int64 __fastcall sub_26CA2E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v3; // r14
  __int64 v4; // rdx
  __int64 v5; // rsi
  __int64 v6; // r9
  int v7; // ebx
  unsigned int v8; // ecx
  __int64 v9; // rdi
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // r12
  char *v15; // rdx
  __int64 v16; // r15
  _QWORD *v17; // rax
  char v18; // dl
  __int64 v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // esi
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 *v25; // rbx
  int v26; // r11d
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rbx
  unsigned __int64 v31; // rax
  __int64 v32; // r13
  int v33; // eax
  unsigned __int8 v34; // r12
  unsigned __int8 v35; // r14
  __int64 v36; // rax
  unsigned int v37; // r13d
  __int64 v38; // r15
  __int64 *v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // r12
  _QWORD *v44; // rax
  char v45; // dl
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rcx
  int v49; // eax
  int v50; // eax
  _QWORD *v51; // rax
  int v52; // edi
  int v53; // ecx
  __int64 v54; // [rsp+0h] [rbp-120h]
  __int64 v55; // [rsp+18h] [rbp-108h]
  __int64 v56; // [rsp+20h] [rbp-100h]
  __int64 v58; // [rsp+30h] [rbp-F0h]
  int v59; // [rsp+3Ch] [rbp-E4h]
  __int64 v60; // [rsp+40h] [rbp-E0h] BYREF
  __int64 *v61; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v62; // [rsp+50h] [rbp-D0h] BYREF
  void *s; // [rsp+58h] [rbp-C8h]
  _BYTE v64[12]; // [rsp+60h] [rbp-C0h]
  unsigned __int8 v65; // [rsp+6Ch] [rbp-B4h]
  char v66; // [rsp+70h] [rbp-B0h] BYREF

  result = a2 + 72;
  v54 = a1 + 1024;
  v56 = *(_QWORD *)(a2 + 80);
  v55 = a2 + 72;
  if ( v56 != a2 + 72 )
  {
    v3 = &v62;
    while ( 1 )
    {
      v62 = 0;
      *(_QWORD *)v64 = 16;
      v4 = v56 - 24;
      *(_DWORD *)&v64[8] = 0;
      if ( !v56 )
        v4 = 0;
      v65 = 1;
      s = &v66;
      v60 = v4;
      v5 = *(unsigned int *)(a1 + 1048);
      if ( !(_DWORD)v5 )
        break;
      v6 = *(_QWORD *)(a1 + 1032);
      v7 = 1;
      v8 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v9 = v6 + 88LL * v8;
      v10 = 0;
      v11 = *(_QWORD *)v9;
      if ( v4 != *(_QWORD *)v9 )
      {
        while ( v11 != -4096 )
        {
          if ( v11 == -8192 && !v10 )
            v10 = (__int64 *)v9;
          v8 = (v5 - 1) & (v7 + v8);
          v9 = v6 + 88LL * v8;
          v11 = *(_QWORD *)v9;
          if ( v4 == *(_QWORD *)v9 )
            goto LABEL_7;
          ++v7;
        }
        if ( !v10 )
          v10 = (__int64 *)v9;
        ++*(_QWORD *)(a1 + 1024);
        v52 = *(_DWORD *)(a1 + 1040);
        v61 = v10;
        v53 = v52 + 1;
        if ( 4 * (v52 + 1) < (unsigned int)(3 * v5) )
        {
          v11 = (unsigned int)v5 >> 3;
          if ( (int)v5 - *(_DWORD *)(a1 + 1044) - v53 > (unsigned int)v11 )
            goto LABEL_84;
          goto LABEL_93;
        }
LABEL_92:
        LODWORD(v5) = 2 * v5;
LABEL_93:
        sub_26C9E50(v54, v5);
        v5 = (__int64)&v60;
        sub_26C3200(v54, &v60, &v61);
        v4 = v60;
        v53 = *(_DWORD *)(a1 + 1040) + 1;
        v10 = v61;
LABEL_84:
        *(_DWORD *)(a1 + 1040) = v53;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 1044);
        *v10 = v4;
        v12 = (__int64)(v10 + 1);
        v10[2] = 0x800000000LL;
        v10[1] = (__int64)(v10 + 3);
        v13 = v65;
        v14 = *(_QWORD *)(v60 + 16);
        if ( !v14 )
          goto LABEL_22;
        goto LABEL_9;
      }
LABEL_7:
      v12 = v9 + 8;
      if ( *(_DWORD *)(v9 + 16) )
        goto LABEL_59;
      v13 = v65;
      v14 = *(_QWORD *)(v60 + 16);
      if ( !v14 )
      {
        ++v62;
        goto LABEL_27;
      }
      do
      {
LABEL_9:
        v15 = *(char **)(v14 + 24);
        if ( (unsigned __int8)(*v15 - 30) <= 0xAu )
        {
          v16 = *((_QWORD *)v15 + 5);
          if ( (_BYTE)v13 )
            goto LABEL_11;
LABEL_18:
          while ( 1 )
          {
            v5 = v16;
            sub_C8CC70((__int64)v3, v16, (__int64)v15, v13, v11, v6);
            v13 = v65;
            if ( v18 )
              break;
            do
            {
LABEL_15:
              v14 = *(_QWORD *)(v14 + 8);
              if ( !v14 )
                goto LABEL_22;
LABEL_16:
              v15 = *(char **)(v14 + 24);
            }
            while ( (unsigned __int8)(*v15 - 30) > 0xAu );
            v16 = *((_QWORD *)v15 + 5);
            if ( (_BYTE)v13 )
            {
LABEL_11:
              v17 = s;
              v5 = *(unsigned int *)&v64[4];
              v15 = (char *)s + 8 * *(unsigned int *)&v64[4];
              if ( s == v15 )
                goto LABEL_53;
              while ( v16 != *v17 )
              {
                if ( v15 == (char *)++v17 )
                {
LABEL_53:
                  if ( *(_DWORD *)&v64[4] < *(_DWORD *)v64 )
                  {
                    v5 = (unsigned int)++*(_DWORD *)&v64[4];
                    *(_QWORD *)v15 = v16;
                    ++v62;
                    goto LABEL_19;
                  }
                  goto LABEL_18;
                }
              }
              goto LABEL_15;
            }
          }
LABEL_19:
          v19 = *(unsigned int *)(v12 + 8);
          if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
          {
            v5 = v12 + 16;
            sub_C8D5F0(v12, (const void *)(v12 + 16), v19 + 1, 8u, v11, v6);
            v19 = *(unsigned int *)(v12 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v12 + 8 * v19) = v16;
          ++*(_DWORD *)(v12 + 8);
          v14 = *(_QWORD *)(v14 + 8);
          v13 = v65;
          if ( !v14 )
            break;
          goto LABEL_16;
        }
        v14 = *(_QWORD *)(v14 + 8);
      }
      while ( v14 );
LABEL_22:
      ++v62;
      if ( (_BYTE)v13 )
        goto LABEL_27;
      v20 = 4 * (*(_DWORD *)&v64[4] - *(_DWORD *)&v64[8]);
      if ( v20 < 0x20 )
        v20 = 32;
      if ( v20 >= *(_DWORD *)v64 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v64);
LABEL_27:
        *(_QWORD *)&v64[4] = 0;
        goto LABEL_28;
      }
      sub_C8C990((__int64)v3, v5);
LABEL_28:
      v21 = *(_DWORD *)(a1 + 1080);
      v22 = a1 + 1056;
      if ( !v21 )
      {
        v61 = 0;
        ++*(_QWORD *)(a1 + 1056);
LABEL_89:
        v21 *= 2;
LABEL_90:
        sub_26C9E50(v22, v21);
        sub_26C3200(v22, &v60, &v61);
        v23 = v60;
        v25 = v61;
        v50 = *(_DWORD *)(a1 + 1072) + 1;
        goto LABEL_71;
      }
      v23 = v60;
      v24 = *(_QWORD *)(a1 + 1064);
      v25 = 0;
      v26 = 1;
      LODWORD(v27) = (v21 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
      v28 = v24 + 88LL * (unsigned int)v27;
      v29 = *(_QWORD *)v28;
      if ( v60 == *(_QWORD *)v28 )
      {
LABEL_30:
        v30 = v28 + 8;
        if ( *(_DWORD *)(v28 + 16) )
          goto LABEL_59;
        goto LABEL_31;
      }
      while ( v29 != -4096 )
      {
        if ( !v25 && v29 == -8192 )
          v25 = (__int64 *)v28;
        v27 = (v21 - 1) & ((_DWORD)v27 + v26);
        v28 = v24 + 88 * v27;
        v29 = *(_QWORD *)v28;
        if ( v60 == *(_QWORD *)v28 )
          goto LABEL_30;
        ++v26;
      }
      if ( !v25 )
        v25 = (__int64 *)v28;
      ++*(_QWORD *)(a1 + 1056);
      v49 = *(_DWORD *)(a1 + 1072);
      v61 = v25;
      v50 = v49 + 1;
      if ( 4 * v50 >= 3 * v21 )
        goto LABEL_89;
      if ( v21 - *(_DWORD *)(a1 + 1076) - v50 <= v21 >> 3 )
        goto LABEL_90;
LABEL_71:
      *(_DWORD *)(a1 + 1072) = v50;
      if ( *v25 != -4096 )
        --*(_DWORD *)(a1 + 1076);
      v51 = v25 + 3;
      *v25 = v23;
      v30 = (__int64)(v25 + 1);
      *(_QWORD *)v30 = v51;
      *(_QWORD *)(v30 + 8) = 0x800000000LL;
LABEL_31:
      v31 = *(_QWORD *)(v60 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v31 != v60 + 48 )
      {
        if ( !v31 )
LABEL_59:
          BUG();
        v32 = v31 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v31 - 24) - 30 <= 0xA )
        {
          v33 = sub_B46E30(v32);
          v34 = v65;
          v59 = v33;
          if ( !v33 )
            goto LABEL_43;
          v58 = (__int64)v3;
          v35 = v65;
          v36 = v32;
          v37 = 0;
          v38 = v36;
          while ( 2 )
          {
            while ( 2 )
            {
              v43 = sub_B46EC0(v38, v37);
              if ( !v35 )
                goto LABEL_49;
              v44 = s;
              v40 = *(unsigned int *)&v64[4];
              v39 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v64[4]);
              if ( s != v39 )
              {
                while ( v43 != *v44 )
                {
                  if ( v39 == ++v44 )
                    goto LABEL_55;
                }
LABEL_41:
                if ( v59 == ++v37 )
                  goto LABEL_42;
                continue;
              }
              break;
            }
LABEL_55:
            if ( *(_DWORD *)&v64[4] < *(_DWORD *)v64 )
            {
              ++*(_DWORD *)&v64[4];
              *v39 = v43;
              v46 = *(unsigned int *)(v30 + 8);
              v48 = *(unsigned int *)(v30 + 12);
              ++v62;
              v47 = v46 + 1;
              if ( v46 + 1 > v48 )
                goto LABEL_57;
            }
            else
            {
LABEL_49:
              sub_C8CC70(v58, v43, (__int64)v39, v40, v41, v42);
              v35 = v65;
              if ( !v45 )
                goto LABEL_41;
              v46 = *(unsigned int *)(v30 + 8);
              v47 = v46 + 1;
              if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(v30 + 12) )
              {
LABEL_57:
                sub_C8D5F0(v30, (const void *)(v30 + 16), v47, 8u, v41, v42);
                v46 = *(unsigned int *)(v30 + 8);
              }
            }
            ++v37;
            *(_QWORD *)(*(_QWORD *)v30 + 8 * v46) = v43;
            ++*(_DWORD *)(v30 + 8);
            v35 = v65;
            if ( v59 == v37 )
            {
LABEL_42:
              v34 = v35;
              v3 = (__int64 *)v58;
              goto LABEL_43;
            }
            continue;
          }
        }
      }
      v34 = v65;
LABEL_43:
      if ( !v34 )
        _libc_free((unsigned __int64)s);
      result = *(_QWORD *)(v56 + 8);
      v56 = result;
      if ( v55 == result )
        return result;
    }
    v61 = 0;
    ++*(_QWORD *)(a1 + 1024);
    goto LABEL_92;
  }
  return result;
}
