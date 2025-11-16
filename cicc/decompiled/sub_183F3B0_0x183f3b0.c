// Function: sub_183F3B0
// Address: 0x183f3b0
//
__int64 *__fastcall sub_183F3B0(size_t a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r13
  size_t v10; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // r8d
  int v15; // r9d
  __int64 *result; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // r10
  unsigned int v21; // r9d
  unsigned __int64 *v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rsi
  unsigned __int64 v26; // rbx
  unsigned int v27; // esi
  int v28; // esi
  int v29; // esi
  __int64 v30; // r9
  unsigned int v31; // ecx
  unsigned __int64 *v32; // rdx
  unsigned __int64 v33; // rdi
  int v34; // eax
  _BYTE *v35; // rax
  _BYTE *v36; // rsi
  int v37; // r13d
  unsigned __int64 v38; // rbx
  char *v39; // r15
  size_t v40; // r14
  char *v41; // rbx
  __int64 v42; // rdi
  __int64 v43; // rsi
  __int64 *v44; // rax
  __int64 *v45; // rsi
  _QWORD *v46; // rcx
  __int64 v47; // rdi
  __int64 v48; // rsi
  int v49; // r8d
  int v50; // eax
  int v51; // esi
  int v52; // esi
  __int64 v53; // r9
  unsigned __int64 *v54; // r10
  int v55; // r8d
  unsigned int v56; // ecx
  unsigned __int64 v57; // rdi
  int v58; // edx
  __int64 *v59; // r8
  int v60; // r8d
  unsigned int v61; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v62; // [rsp+20h] [rbp-C0h]
  __int64 v63; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v65; // [rsp+48h] [rbp-98h] BYREF
  char v66[8]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v67; // [rsp+58h] [rbp-88h]
  __int64 v68; // [rsp+68h] [rbp-78h]
  char v69[8]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v70; // [rsp+78h] [rbp-68h]
  __int64 v71; // [rsp+88h] [rbp-58h]
  int v72; // [rsp+90h] [rbp-50h] BYREF
  __int64 v73; // [rsp+98h] [rbp-48h]
  __int64 v74; // [rsp+A0h] [rbp-40h]
  __int64 v75; // [rsp+A8h] [rbp-38h]

  v5 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 24;
  v7 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 72;
  v8 = a2 & 4;
  if ( (_DWORD)v8 )
    v7 = v6;
  v9 = *(_QWORD *)v7;
  if ( *(_BYTE *)(*(_QWORD *)v7 + 16LL) )
  {
    v65 = v5;
    v44 = *(__int64 **)(a1 + 112);
    if ( *(__int64 **)(a1 + 120) == v44 )
    {
      v12 = *(unsigned int *)(a1 + 132);
      v45 = &v44[v12];
      v10 = v12;
      if ( v44 != v45 )
      {
        v46 = 0;
        while ( 1 )
        {
          v12 = *v44;
          if ( v5 == *v44 )
            break;
          if ( v12 == -2 )
            v46 = v44;
          if ( v45 == ++v44 )
          {
            if ( !v46 )
              goto LABEL_76;
            *v46 = v5;
            --*(_DWORD *)(a1 + 136);
            ++*(_QWORD *)(a1 + 104);
            break;
          }
        }
LABEL_5:
        result = *(__int64 **)v5;
        if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) )
        {
          v35 = *(_BYTE **)(a1 + 56);
          v36 = *(_BYTE **)(a1 + 48);
          v37 = *(_DWORD *)(a1 + 40);
          v38 = v35 - v36;
          if ( v35 == v36 )
          {
            v40 = 0;
            v39 = 0;
          }
          else
          {
            if ( v38 > 0x7FFFFFFFFFFFFFF8LL )
              sub_4261EA(v10, v36, v12);
            v39 = (char *)sub_22077B0(v38);
            v35 = *(_BYTE **)(a1 + 56);
            v36 = *(_BYTE **)(a1 + 48);
            v40 = v35 - v36;
          }
          v41 = &v39[v38];
          if ( v35 != v36 )
            memmove(v39, v36, v40);
          result = sub_183BD80(a3, (__int64 *)&v65);
          v42 = result[2];
          v43 = result[4];
          *((_DWORD *)result + 2) = v37;
          result[2] = (__int64)v39;
          result[3] = (__int64)&v39[v40];
          result[4] = (__int64)v41;
          if ( v42 )
            return (__int64 *)j_j___libc_free_0(v42, v43 - v42);
        }
        return result;
      }
LABEL_76:
      if ( (unsigned int)v10 < *(_DWORD *)(a1 + 128) )
      {
        v10 = (unsigned int)(v10 + 1);
        *(_DWORD *)(a1 + 132) = v10;
        *v45 = v5;
        ++*(_QWORD *)(a1 + 104);
        goto LABEL_5;
      }
    }
    v10 = a1 + 104;
    sub_16CCBA0(a1 + 104, v5);
    goto LABEL_5;
  }
  v10 = *(_QWORD *)v7;
  v65 = v5;
  if ( !(unsigned __int8)sub_387E010(v10, v8) )
    goto LABEL_5;
  v17 = *(_QWORD *)(v9 + 80);
  if ( v17 )
    v17 -= 24;
  sub_183B530((__int64)a4, v17, v12, v13, v14, v15);
  if ( (*(_BYTE *)(v9 + 18) & 1) != 0 )
  {
    sub_15E08E0(v9, v17);
    v18 = *(_QWORD *)(v9 + 88);
    v63 = v18 + 40LL * *(_QWORD *)(v9 + 96);
    if ( (*(_BYTE *)(v9 + 18) & 1) != 0 )
    {
      sub_15E08E0(v9, v17);
      v18 = *(_QWORD *)(v9 + 88);
    }
  }
  else
  {
    v18 = *(_QWORD *)(v9 + 88);
    v63 = v18 + 40LL * *(_QWORD *)(v9 + 96);
  }
  if ( v63 != v18 )
  {
    v19 = v18;
    v62 = v5;
    while ( 1 )
    {
      v26 = v19 & 0xFFFFFFFFFFFFFFF9LL;
      sub_183C910(
        (__int64)v69,
        a4,
        *(_QWORD *)(v62 + 24 * (*(unsigned int *)(v19 + 32) - (unsigned __int64)(*(_DWORD *)(v62 + 20) & 0xFFFFFFF)))
      & 0xFFFFFFFFFFFFFFF9LL);
      sub_183C910((__int64)v66, a4, v19 & 0xFFFFFFFFFFFFFFF9LL);
      sub_183EA00((const char *)&v72, a1, (__int64)v66, (__int64)v69);
      v27 = *(_DWORD *)(a3 + 24);
      if ( !v27 )
        break;
      v20 = *(_QWORD *)(a3 + 8);
      v21 = (v27 - 1) & (v26 ^ (v26 >> 9));
      v22 = (unsigned __int64 *)(v20 + 40LL * v21);
      v23 = *v22;
      if ( v26 == *v22 )
      {
        v24 = v22[2];
        v25 = v22[4] - v24;
        goto LABEL_15;
      }
      v49 = 1;
      v32 = 0;
      while ( 1 )
      {
        if ( v23 == -2 )
        {
          if ( !v32 )
            v32 = v22;
          v50 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v34 = v50 + 1;
          if ( 4 * v34 < 3 * v27 )
          {
            if ( v27 - *(_DWORD *)(a3 + 20) - v34 > v27 >> 3 )
              goto LABEL_27;
            v61 = v26 ^ (v26 >> 9);
            sub_183B620(a3, v27);
            v51 = *(_DWORD *)(a3 + 24);
            if ( v51 )
            {
              v52 = v51 - 1;
              v53 = *(_QWORD *)(a3 + 8);
              v54 = 0;
              v55 = 1;
              v56 = v52 & v61;
              v32 = (unsigned __int64 *)(v53 + 40LL * (v52 & v61));
              v57 = *v32;
              v34 = *(_DWORD *)(a3 + 16) + 1;
              if ( v26 != *v32 )
              {
                while ( v57 != -2 )
                {
                  if ( !v54 && v57 == -16 )
                    v54 = v32;
                  v56 = v52 & (v55 + v56);
                  v32 = (unsigned __int64 *)(v53 + 40LL * v56);
                  v57 = *v32;
                  if ( v26 == *v32 )
                    goto LABEL_27;
                  ++v55;
                }
LABEL_67:
                if ( v54 )
                  v32 = v54;
              }
LABEL_27:
              *(_DWORD *)(a3 + 16) = v34;
              if ( *v32 != -2 )
                --*(_DWORD *)(a3 + 20);
              *v32 = v26;
              *((_DWORD *)v32 + 2) = v72;
              v32[2] = v73;
              v32[3] = v74;
              v32[4] = v75;
              goto LABEL_18;
            }
LABEL_93:
            ++*(_DWORD *)(a3 + 16);
            BUG();
          }
LABEL_25:
          sub_183B620(a3, 2 * v27);
          v28 = *(_DWORD *)(a3 + 24);
          if ( v28 )
          {
            v29 = v28 - 1;
            v30 = *(_QWORD *)(a3 + 8);
            v31 = v29 & (v26 ^ (v26 >> 9));
            v32 = (unsigned __int64 *)(v30 + 40LL * v31);
            v33 = *v32;
            v34 = *(_DWORD *)(a3 + 16) + 1;
            if ( v26 != *v32 )
            {
              v60 = 1;
              v54 = 0;
              while ( v33 != -2 )
              {
                if ( v33 == -16 && !v54 )
                  v54 = v32;
                v31 = v29 & (v60 + v31);
                v32 = (unsigned __int64 *)(v30 + 40LL * v31);
                v33 = *v32;
                if ( v26 == *v32 )
                  goto LABEL_27;
                ++v60;
              }
              goto LABEL_67;
            }
            goto LABEL_27;
          }
          goto LABEL_93;
        }
        if ( v32 || v23 != -16 )
          v22 = v32;
        v58 = v49 + 1;
        v21 = (v27 - 1) & (v49 + v21);
        v59 = (__int64 *)(v20 + 40LL * v21);
        v23 = *v59;
        if ( v26 == *v59 )
          break;
        v49 = v58;
        v32 = v22;
        v22 = (unsigned __int64 *)(v20 + 40LL * v21);
      }
      v24 = v59[2];
      v22 = (unsigned __int64 *)(v20 + 40LL * v21);
      v25 = v59[4] - v24;
LABEL_15:
      *((_DWORD *)v22 + 2) = v72;
      v22[2] = v73;
      v22[3] = v74;
      v22[4] = v75;
      v73 = 0;
      v74 = 0;
      v75 = 0;
      if ( v24 )
      {
        j_j___libc_free_0(v24, v25);
        if ( v73 )
          j_j___libc_free_0(v73, v75 - v73);
      }
LABEL_18:
      if ( v67 )
        j_j___libc_free_0(v67, v68 - v67);
      if ( v70 )
        j_j___libc_free_0(v70, v71 - v70);
      v19 += 40;
      if ( v63 == v19 )
      {
        v5 = v62;
        goto LABEL_50;
      }
    }
    ++*(_QWORD *)a3;
    goto LABEL_25;
  }
LABEL_50:
  result = *(__int64 **)v5;
  if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) )
  {
    sub_183C910((__int64)v69, a4, v9 & 0xFFFFFFFFFFFFFFF9LL | 2);
    sub_183C910((__int64)v66, a4, v65);
    sub_183EA00((const char *)&v72, a1, (__int64)v66, (__int64)v69);
    result = sub_183BD80(a3, (__int64 *)&v65);
    v47 = result[2];
    v48 = result[4];
    *((_DWORD *)result + 2) = v72;
    result[2] = v73;
    result[3] = v74;
    result[4] = v75;
    v73 = 0;
    v74 = 0;
    v75 = 0;
    if ( v47 )
    {
      result = (__int64 *)j_j___libc_free_0(v47, v48 - v47);
      if ( v73 )
        result = (__int64 *)j_j___libc_free_0(v73, v75 - v73);
    }
    if ( v67 )
      result = (__int64 *)j_j___libc_free_0(v67, v68 - v67);
    v42 = v70;
    if ( v70 )
    {
      v43 = v71;
      return (__int64 *)j_j___libc_free_0(v42, v43 - v42);
    }
  }
  return result;
}
