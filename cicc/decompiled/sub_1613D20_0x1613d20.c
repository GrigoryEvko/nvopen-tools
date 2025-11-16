// Function: sub_1613D20
// Address: 0x1613d20
//
__int64 *__fastcall sub_1613D20(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 *result; // rax
  unsigned int v7; // esi
  __int64 v8; // r11
  __int64 v9; // rdi
  unsigned int v10; // ebx
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // r13
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rax
  __int64 *v27; // r13
  __int64 v28; // r12
  __int64 *v29; // rbx
  __int64 v30; // r14
  unsigned int v31; // esi
  __int64 v32; // r8
  unsigned int v33; // ecx
  _QWORD *v34; // rax
  __int64 v35; // rdi
  int v36; // r11d
  __int64 *v37; // r10
  int v38; // edi
  int v39; // edx
  int v40; // r11d
  int v41; // r11d
  __int64 v42; // r9
  unsigned int v43; // ecx
  __int64 v44; // r8
  int v45; // edi
  __int64 *v46; // rsi
  int v47; // r11d
  _QWORD *v48; // r10
  int v49; // edi
  int v50; // edi
  int v51; // r10d
  int v52; // r10d
  __int64 v53; // r8
  __int64 *v54; // rcx
  unsigned int v55; // ebx
  int v56; // esi
  __int64 v57; // rdi
  int v58; // r8d
  int v59; // r8d
  __int64 v60; // r9
  unsigned int v61; // edx
  __int64 v62; // r11
  int v63; // esi
  _QWORD *v64; // rcx
  int v65; // r9d
  int v66; // r9d
  __int64 v67; // r10
  unsigned int v68; // edx
  int v69; // esi
  __int64 v70; // r8
  __int64 v71; // [rsp+8h] [rbp-158h]
  __int64 v72; // [rsp+8h] [rbp-158h]
  __int64 v73; // [rsp+10h] [rbp-150h]
  unsigned int v74; // [rsp+1Ch] [rbp-144h]
  __int64 *v75; // [rsp+30h] [rbp-130h]
  __int64 v77; // [rsp+40h] [rbp-120h]
  unsigned int v78; // [rsp+40h] [rbp-120h]
  __int64 *v79; // [rsp+48h] [rbp-118h]
  _BYTE *v80; // [rsp+50h] [rbp-110h] BYREF
  __int64 v81; // [rsp+58h] [rbp-108h]
  _BYTE v82[96]; // [rsp+60h] [rbp-100h] BYREF
  _BYTE *v83; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v84; // [rsp+C8h] [rbp-98h]
  _BYTE v85[144]; // [rsp+D0h] [rbp-90h] BYREF

  v5 = *(_QWORD *)(a4 + 8);
  v74 = 0;
  if ( v5 )
    v74 = *(_DWORD *)(*(_QWORD *)(v5 + 24) + 400LL);
  result = &a2[a3];
  v79 = a2;
  v73 = a1 + 192;
  v75 = result;
  if ( result != a2 )
  {
    while ( 1 )
    {
      v7 = *(_DWORD *)(a1 + 216);
      v8 = *v79;
      v77 = *v79;
      if ( !v7 )
        break;
      v9 = *(_QWORD *)(a1 + 200);
      v10 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
      v11 = (v7 - 1) & v10;
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v8 != *v12 )
      {
        v36 = 1;
        v37 = 0;
        while ( v13 != -8 )
        {
          if ( v13 == -16 && !v37 )
            v37 = v12;
          v11 = (v7 - 1) & (v36 + v11);
          v12 = (__int64 *)(v9 + 16LL * v11);
          v13 = *v12;
          if ( v77 == *v12 )
            goto LABEL_6;
          ++v36;
        }
        v38 = *(_DWORD *)(a1 + 208);
        if ( v37 )
          v12 = v37;
        ++*(_QWORD *)(a1 + 192);
        v39 = v38 + 1;
        if ( 4 * (v38 + 1) < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(a1 + 212) - v39 <= v7 >> 3 )
          {
            sub_1613260(v73, v7);
            v51 = *(_DWORD *)(a1 + 216);
            if ( !v51 )
            {
LABEL_120:
              ++*(_DWORD *)(a1 + 208);
              BUG();
            }
            v52 = v51 - 1;
            v53 = *(_QWORD *)(a1 + 200);
            v54 = 0;
            v55 = v52 & v10;
            v39 = *(_DWORD *)(a1 + 208) + 1;
            v56 = 1;
            v12 = (__int64 *)(v53 + 16LL * v55);
            v57 = *v12;
            if ( v77 != *v12 )
            {
              while ( v57 != -8 )
              {
                if ( v57 == -16 && !v54 )
                  v54 = v12;
                v55 = v52 & (v56 + v55);
                v12 = (__int64 *)(v53 + 16LL * v55);
                v57 = *v12;
                if ( v77 == *v12 )
                  goto LABEL_51;
                ++v56;
              }
              if ( v54 )
                v12 = v54;
            }
          }
LABEL_51:
          *(_DWORD *)(a1 + 208) = v39;
          if ( *v12 != -8 )
            --*(_DWORD *)(a1 + 212);
          v12[1] = 0;
          *v12 = v77;
          goto LABEL_6;
        }
LABEL_55:
        sub_1613260(v73, 2 * v7);
        v40 = *(_DWORD *)(a1 + 216);
        if ( !v40 )
          goto LABEL_120;
        v41 = v40 - 1;
        v42 = *(_QWORD *)(a1 + 200);
        v39 = *(_DWORD *)(a1 + 208) + 1;
        v43 = v41 & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
        v12 = (__int64 *)(v42 + 16LL * v43);
        v44 = *v12;
        if ( v77 != *v12 )
        {
          v45 = 1;
          v46 = 0;
          while ( v44 != -8 )
          {
            if ( !v46 && v44 == -16 )
              v46 = v12;
            v43 = v41 & (v45 + v43);
            v12 = (__int64 *)(v42 + 16LL * v43);
            v44 = *v12;
            if ( v77 == *v12 )
              goto LABEL_51;
            ++v45;
          }
          if ( v46 )
            v12 = v46;
        }
        goto LABEL_51;
      }
LABEL_6:
      v12[1] = a4;
      if ( v77 == a4 )
        goto LABEL_7;
      v14 = sub_16135E0(a1, v77);
      v15 = *(__int64 **)(v14 + 80);
      v16 = *(unsigned int *)(v14 + 88);
      v80 = v82;
      v17 = &v15[v16];
      v81 = 0xC00000000LL;
      v83 = v85;
      v84 = 0xC00000000LL;
      if ( v15 == v17 )
      {
        v21 = v82;
        v22 = 0;
        goto LABEL_18;
      }
      do
      {
        while ( 1 )
        {
          v18 = sub_160EA80(a1, *v15);
          v19 = *(_QWORD *)(*(_QWORD *)(v18 + 8) + 24LL);
          if ( v74 == *(_DWORD *)(v19 + 400) )
          {
            v25 = (unsigned int)v81;
            if ( (unsigned int)v81 >= HIDWORD(v81) )
            {
              v72 = v18;
              sub_16CD150(&v80, v82, 0, 8);
              v25 = (unsigned int)v81;
              v18 = v72;
            }
            *(_QWORD *)&v80[8 * v25] = v18;
            LODWORD(v81) = v81 + 1;
            goto LABEL_11;
          }
          if ( v74 > *(_DWORD *)(v19 + 400) )
            break;
LABEL_11:
          if ( v17 == ++v15 )
            goto LABEL_17;
        }
        v20 = (unsigned int)v84;
        if ( (unsigned int)v84 >= HIDWORD(v84) )
        {
          v71 = v18;
          sub_16CD150(&v83, v85, 0, 8);
          v20 = (unsigned int)v84;
          v18 = v71;
        }
        ++v15;
        *(_QWORD *)&v83[8 * v20] = v18;
        LODWORD(v84) = v84 + 1;
      }
      while ( v17 != v15 );
LABEL_17:
      v21 = v80;
      v22 = (unsigned int)v81;
LABEL_18:
      sub_1613D20(a1, v21, v22, a4);
      v23 = *(_QWORD *)(a4 + 8);
      if ( v23 )
      {
        v24 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v23 + 24) + 16LL))(*(_QWORD *)(v23 + 24));
        sub_1613D20(a1, v83, (unsigned int)v84, v24);
      }
      if ( !*(_DWORD *)(a1 + 208) )
        goto LABEL_21;
      v26 = *(__int64 **)(a1 + 200);
      v27 = &v26[2 * *(unsigned int *)(a1 + 216)];
      if ( v26 == v27 )
        goto LABEL_21;
      while ( 1 )
      {
        v28 = *v26;
        v29 = v26;
        if ( *v26 != -16 && v28 != -8 )
          break;
        v26 += 2;
        if ( v27 == v26 )
          goto LABEL_21;
      }
      if ( v26 == v27 )
        goto LABEL_21;
      v30 = v77;
      if ( v77 == v26[1] )
      {
LABEL_42:
        v31 = *(_DWORD *)(a1 + 216);
        if ( v31 )
        {
          v32 = *(_QWORD *)(a1 + 200);
          v33 = (v31 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v34 = (_QWORD *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( v28 == *v34 )
          {
LABEL_44:
            v34[1] = a4;
            goto LABEL_35;
          }
          v47 = 1;
          v48 = 0;
          while ( v35 != -8 )
          {
            if ( !v48 && v35 == -16 )
              v48 = v34;
            v33 = (v31 - 1) & (v47 + v33);
            v34 = (_QWORD *)(v32 + 16LL * v33);
            v35 = *v34;
            if ( v28 == *v34 )
              goto LABEL_44;
            ++v47;
          }
          v49 = *(_DWORD *)(a1 + 208);
          if ( v48 )
            v34 = v48;
          ++*(_QWORD *)(a1 + 192);
          v50 = v49 + 1;
          if ( 4 * v50 < 3 * v31 )
          {
            if ( v31 - *(_DWORD *)(a1 + 212) - v50 <= v31 >> 3 )
            {
              v78 = ((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4);
              sub_1613260(v73, v31);
              v65 = *(_DWORD *)(a1 + 216);
              if ( !v65 )
              {
LABEL_119:
                ++*(_DWORD *)(a1 + 208);
                BUG();
              }
              v66 = v65 - 1;
              v67 = *(_QWORD *)(a1 + 200);
              v64 = 0;
              v68 = v66 & v78;
              v50 = *(_DWORD *)(a1 + 208) + 1;
              v69 = 1;
              v34 = (_QWORD *)(v67 + 16LL * (v66 & v78));
              v70 = *v34;
              if ( *v34 != v28 )
              {
                while ( v70 != -8 )
                {
                  if ( !v64 && v70 == -16 )
                    v64 = v34;
                  v68 = v66 & (v69 + v68);
                  v34 = (_QWORD *)(v67 + 16LL * v68);
                  v70 = *v34;
                  if ( v28 == *v34 )
                    goto LABEL_69;
                  ++v69;
                }
                goto LABEL_83;
              }
            }
            goto LABEL_69;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 192);
        }
        sub_1613260(v73, 2 * v31);
        v58 = *(_DWORD *)(a1 + 216);
        if ( !v58 )
          goto LABEL_119;
        v59 = v58 - 1;
        v60 = *(_QWORD *)(a1 + 200);
        v61 = v59 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v50 = *(_DWORD *)(a1 + 208) + 1;
        v34 = (_QWORD *)(v60 + 16LL * v61);
        v62 = *v34;
        if ( v28 != *v34 )
        {
          v63 = 1;
          v64 = 0;
          while ( v62 != -8 )
          {
            if ( v62 == -16 && !v64 )
              v64 = v34;
            v61 = v59 & (v63 + v61);
            v34 = (_QWORD *)(v60 + 16LL * v61);
            v62 = *v34;
            if ( v28 == *v34 )
              goto LABEL_69;
            ++v63;
          }
LABEL_83:
          if ( v64 )
            v34 = v64;
        }
LABEL_69:
        *(_DWORD *)(a1 + 208) = v50;
        if ( *v34 != -8 )
          --*(_DWORD *)(a1 + 212);
        *v34 = v28;
        v34[1] = 0;
        goto LABEL_44;
      }
LABEL_35:
      while ( 1 )
      {
        v29 += 2;
        if ( v29 == v27 )
          break;
        while ( 1 )
        {
          v28 = *v29;
          if ( *v29 != -8 && v28 != -16 )
            break;
          v29 += 2;
          if ( v27 == v29 )
            goto LABEL_21;
        }
        if ( v29 == v27 )
          break;
        if ( v30 == v29[1] )
          goto LABEL_42;
      }
LABEL_21:
      if ( v83 != v85 )
        _libc_free((unsigned __int64)v83);
      if ( v80 != v82 )
        _libc_free((unsigned __int64)v80);
LABEL_7:
      result = ++v79;
      if ( v75 == v79 )
        return result;
    }
    ++*(_QWORD *)(a1 + 192);
    goto LABEL_55;
  }
  return result;
}
