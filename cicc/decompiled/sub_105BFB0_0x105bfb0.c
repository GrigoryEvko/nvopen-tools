// Function: sub_105BFB0
// Address: 0x105bfb0
//
__int64 __fastcall sub_105BFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r13
  bool v8; // zf
  __int64 v9; // r12
  _QWORD *v10; // rax
  int v11; // r11d
  unsigned int v12; // esi
  int v13; // r15d
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // r10d
  __int64 *v17; // rdx
  unsigned int v18; // ecx
  _QWORD *v19; // rax
  __int64 v20; // rdi
  _DWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 *v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rbx
  unsigned int v27; // r15d
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // r8
  unsigned int v31; // r9d
  __int64 v32; // rbx
  _QWORD *v33; // rdi
  __int64 v34; // rsi
  _QWORD *v35; // rax
  int v36; // r9d
  __int64 v37; // r9
  _QWORD *v38; // rax
  _QWORD *v39; // rdx
  __int64 result; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 *v43; // r11
  int v44; // eax
  unsigned int v45; // edx
  __int64 *v46; // rsi
  __int64 v47; // rdi
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 *v50; // rax
  int v51; // esi
  int v52; // edi
  int v53; // ebx
  int v54; // ebx
  __int64 v55; // r10
  __int64 v56; // rsi
  int v57; // ecx
  __int64 *v58; // rax
  int v59; // eax
  int v60; // r10d
  int v61; // ecx
  unsigned int v62; // ebx
  __int64 v63; // rsi
  __int64 v66; // [rsp+20h] [rbp-90h]
  int v67; // [rsp+20h] [rbp-90h]
  int v68; // [rsp+20h] [rbp-90h]
  __int64 v69; // [rsp+28h] [rbp-88h]
  int v70; // [rsp+28h] [rbp-88h]
  int v71; // [rsp+28h] [rbp-88h]
  __int64 v72; // [rsp+38h] [rbp-78h] BYREF
  _BYTE *v73; // [rsp+40h] [rbp-70h] BYREF
  __int64 v74; // [rsp+48h] [rbp-68h]
  _BYTE v75[96]; // [rsp+50h] [rbp-60h] BYREF

  v6 = a3;
  v7 = a4;
  v73 = v75;
  v8 = *(_BYTE *)(a4 + 28) == 0;
  v74 = 0x600000000LL;
  v9 = **(_QWORD **)(a3 + 8);
  if ( v8 )
    goto LABEL_39;
  v10 = *(_QWORD **)(a4 + 8);
  a4 = *(unsigned int *)(a4 + 20);
  a3 = (__int64)&v10[a4];
  if ( v10 == (_QWORD *)a3 )
  {
LABEL_38:
    if ( (unsigned int)a4 >= *(_DWORD *)(v7 + 16) )
    {
LABEL_39:
      sub_C8CC70(v7, v9, a3, a4, a5, a6);
      goto LABEL_6;
    }
    *(_DWORD *)(v7 + 20) = a4 + 1;
    *(_QWORD *)a3 = v9;
    ++*(_QWORD *)v7;
  }
  else
  {
    while ( v9 != *v10 )
    {
      if ( (_QWORD *)a3 == ++v10 )
        goto LABEL_38;
    }
  }
LABEL_6:
  v11 = *(_DWORD *)(v6 + 16);
  v12 = *(_DWORD *)(a1 + 88);
  v13 = *(_DWORD *)(a1 + 8);
  v69 = a1 + 64;
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_68;
  }
  v14 = *(_QWORD *)(a1 + 72);
  v15 = v12 - 1;
  v16 = 1;
  v17 = 0;
  v18 = v15 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v19 = (_QWORD *)(v14 + 16LL * v18);
  v20 = *v19;
  if ( v9 == *v19 )
  {
LABEL_8:
    v21 = v19 + 1;
    goto LABEL_9;
  }
  while ( v20 != -4096 )
  {
    if ( !v17 && v20 == -8192 )
      v17 = v19;
    v18 = v15 & (v16 + v18);
    v19 = (_QWORD *)(v14 + 16LL * v18);
    v20 = *v19;
    if ( v9 == *v19 )
      goto LABEL_8;
    ++v16;
  }
  if ( !v17 )
    v17 = v19;
  ++*(_QWORD *)(a1 + 64);
  v52 = *(_DWORD *)(a1 + 80) + 1;
  if ( 4 * v52 >= 3 * v12 )
  {
LABEL_68:
    v67 = v11;
    sub_A4A350(v69, 2 * v12);
    v53 = *(_DWORD *)(a1 + 88);
    if ( v53 )
    {
      v54 = v53 - 1;
      v55 = *(_QWORD *)(a1 + 72);
      v11 = v67;
      v15 = v54 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v52 = *(_DWORD *)(a1 + 80) + 1;
      v17 = (__int64 *)(v55 + 16 * v15);
      v56 = *v17;
      if ( v9 == *v17 )
        goto LABEL_64;
      v57 = 1;
      v58 = 0;
      while ( v56 != -4096 )
      {
        if ( v56 == -8192 && !v58 )
          v58 = v17;
        v14 = (unsigned int)(v57 + 1);
        v15 = v54 & (unsigned int)(v57 + v15);
        v17 = (__int64 *)(v55 + 16LL * (unsigned int)v15);
        v56 = *v17;
        if ( v9 == *v17 )
          goto LABEL_64;
        ++v57;
      }
LABEL_72:
      if ( v58 )
        v17 = v58;
      goto LABEL_64;
    }
LABEL_88:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
  if ( v12 - *(_DWORD *)(a1 + 84) - v52 <= v12 >> 3 )
  {
    v68 = v11;
    sub_A4A350(v69, v12);
    v59 = *(_DWORD *)(a1 + 88);
    if ( v59 )
    {
      v60 = v59 - 1;
      v15 = *(_QWORD *)(a1 + 72);
      v61 = 1;
      v62 = (v59 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v11 = v68;
      v52 = *(_DWORD *)(a1 + 80) + 1;
      v58 = 0;
      v17 = (__int64 *)(v15 + 16LL * v62);
      v63 = *v17;
      if ( v9 == *v17 )
        goto LABEL_64;
      while ( v63 != -4096 )
      {
        if ( !v58 && v63 == -8192 )
          v58 = v17;
        v14 = (unsigned int)(v61 + 1);
        v62 = v60 & (v61 + v62);
        v17 = (__int64 *)(v15 + 16LL * v62);
        v63 = *v17;
        if ( v9 == *v17 )
          goto LABEL_64;
        ++v61;
      }
      goto LABEL_72;
    }
    goto LABEL_88;
  }
LABEL_64:
  *(_DWORD *)(a1 + 80) = v52;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v17 = v9;
  v21 = v17 + 1;
  *((_DWORD *)v17 + 2) = 0;
LABEL_9:
  *v21 = v13;
  v22 = *(unsigned int *)(a1 + 8);
  if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v71 = v11;
    sub_C8D5F0(a1, (const void *)(a1 + 16), v22 + 1, 8u, v14, v15);
    v22 = *(unsigned int *)(a1 + 8);
    v11 = v71;
  }
  v23 = a1;
  v24 = *(__int64 **)a1;
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v22) = v9;
  ++*(_DWORD *)(a1 + 8);
  if ( v11 != 1 )
    goto LABEL_12;
  if ( !*(_BYTE *)(a1 + 124) )
  {
LABEL_52:
    sub_C8CC70(a1 + 96, v9, (__int64)v24, v23, v14, v15);
    goto LABEL_12;
  }
  v50 = *(__int64 **)(a1 + 104);
  v23 = *(unsigned int *)(a1 + 116);
  v24 = &v50[v23];
  if ( v50 == v24 )
  {
LABEL_44:
    if ( (unsigned int)v23 < *(_DWORD *)(a1 + 112) )
    {
      *(_DWORD *)(a1 + 116) = v23 + 1;
      *v24 = v9;
      ++*(_QWORD *)(a1 + 96);
      goto LABEL_12;
    }
    goto LABEL_52;
  }
  while ( v9 != *v50 )
  {
    if ( v24 == ++v50 )
      goto LABEL_44;
  }
LABEL_12:
  v25 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v25 != v9 + 48 )
  {
    if ( !v25 )
      BUG();
    v26 = v25 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v25 - 24) - 30 <= 0xA )
    {
      v70 = sub_B46E30(v25 - 24);
      if ( v70 )
      {
        v66 = v9;
        v27 = 0;
        v28 = v6;
        v29 = v26;
        while ( 1 )
        {
          while ( 1 )
          {
            v72 = sub_B46EC0(v29, v27);
            v32 = v72;
            if ( *(_DWORD *)(v28 + 72) )
            {
              v41 = *(unsigned int *)(v28 + 80);
              v42 = *(_QWORD *)(v28 + 64);
              v43 = (__int64 *)(v42 + 8 * v41);
              if ( !(_DWORD)v41 )
                goto LABEL_25;
              v44 = v41 - 1;
              v45 = (v41 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
              v46 = (__int64 *)(v42 + 8LL * v45);
              v47 = *v46;
              if ( *v46 != v72 )
              {
                v51 = 1;
                while ( v47 != -4096 )
                {
                  v30 = (unsigned int)(v51 + 1);
                  v45 = v44 & (v51 + v45);
                  v46 = (__int64 *)(v42 + 8LL * v45);
                  v47 = *v46;
                  if ( v72 == *v46 )
                    goto LABEL_32;
                  v51 = v30;
                }
                goto LABEL_25;
              }
LABEL_32:
              LOBYTE(v44) = v43 == v46;
              LOBYTE(v31) = v66 == v72;
              v37 = v44 | v31;
            }
            else
            {
              v33 = *(_QWORD **)(v28 + 88);
              v34 = (__int64)&v33[*(unsigned int *)(v28 + 96)];
              v35 = sub_1055FB0(v33, v34, &v72);
              LOBYTE(v35) = v34 == (_QWORD)v35;
              v37 = (unsigned int)v35 | v36;
            }
            if ( !(_BYTE)v37 )
              break;
LABEL_25:
            if ( ++v27 == v70 )
              goto LABEL_26;
          }
          if ( *(_BYTE *)(v7 + 28) )
          {
            v38 = *(_QWORD **)(v7 + 8);
            v39 = &v38[*(unsigned int *)(v7 + 20)];
            if ( v38 == v39 )
              goto LABEL_34;
            while ( v32 != *v38 )
            {
              if ( v39 == ++v38 )
                goto LABEL_34;
            }
            goto LABEL_25;
          }
          if ( sub_C8CA60(v7, v32) )
            goto LABEL_25;
LABEL_34:
          v48 = (unsigned int)v74;
          v49 = (unsigned int)v74 + 1LL;
          if ( v49 > HIDWORD(v74) )
          {
            sub_C8D5F0((__int64)&v73, v75, v49, 8u, v30, v37);
            v48 = (unsigned int)v74;
          }
          ++v27;
          *(_QWORD *)&v73[8 * v48] = v32;
          LODWORD(v74) = v74 + 1;
          if ( v27 == v70 )
          {
LABEL_26:
            v6 = v28;
            break;
          }
        }
      }
    }
  }
  result = sub_105C640(a1, &v73, a2, v6, v7);
  if ( v73 != v75 )
    return _libc_free(v73, &v73);
  return result;
}
