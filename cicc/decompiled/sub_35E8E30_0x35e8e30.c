// Function: sub_35E8E30
// Address: 0x35e8e30
//
_DWORD *__fastcall sub_35E8E30(__int64 a1, unsigned int a2, _DWORD *a3)
{
  __int64 v4; // rbx
  _DWORD *result; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  _QWORD *v17; // r14
  _QWORD *v18; // r15
  int v19; // r13d
  int v20; // ebx
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // r15
  int v26; // r14d
  __int64 v27; // r9
  __int64 *v28; // rcx
  unsigned int v29; // edx
  unsigned int v30; // r8d
  _QWORD *v31; // rax
  __int64 v32; // rdi
  int v33; // eax
  int v34; // eax
  int v35; // eax
  unsigned __int64 v36; // rsi
  int v37; // r15d
  __int64 v38; // rax
  int v39; // eax
  int v40; // r8d
  int v41; // esi
  int v42; // esi
  __int64 v43; // r8
  __int64 v44; // rdx
  __int64 v45; // rdi
  int v46; // r11d
  __int64 *v47; // r10
  int v48; // esi
  int v49; // esi
  int v50; // r11d
  __int64 v51; // r8
  __int64 v52; // rdx
  __int64 v53; // rdi
  _DWORD *v55; // [rsp+10h] [rbp-50h]
  __int64 v56; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+28h] [rbp-38h]
  unsigned int v59; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 24);
  result = (_DWORD *)sub_2E311E0(v4);
  v6 = *(_QWORD *)(v4 + 56);
  v55 = result;
  if ( (_DWORD *)v6 != result )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(a1 + 72);
      v8 = *(_DWORD *)(v7 + 960);
      v9 = *(_QWORD *)(v7 + 944);
      if ( !v8 )
        goto LABEL_71;
      v10 = v8 - 1;
      v11 = v10 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v6 != *v12 )
      {
        v39 = 1;
        while ( v13 != -4096 )
        {
          v40 = v39 + 1;
          v11 = v10 & (v39 + v11);
          v12 = (__int64 *)(v9 + 16LL * v11);
          v13 = *v12;
          if ( *v12 == v6 )
            goto LABEL_4;
          v39 = v40;
        }
LABEL_71:
        BUG();
      }
LABEL_4:
      v14 = v12[1];
      v15 = *(_QWORD *)(v14 + 120);
      v16 = 16LL * *(unsigned int *)(v14 + 128);
      v17 = (_QWORD *)(v15 + v16);
      if ( v15 != v15 + v16 )
        break;
      v35 = sub_35E72A0(a1, v6);
      if ( v35 )
      {
        v36 = sub_2EBEE10(*(_QWORD *)(a1 + 64), v35);
        if ( *(_QWORD *)(v36 + 24) == *(_QWORD *)(a1 + 24) )
        {
          v19 = 0x7FFFFFFF;
          goto LABEL_39;
        }
      }
LABEL_37:
      v19 = *a3 - 1;
LABEL_13:
      v23 = sub_35E86F0(a1, v6);
      v24 = *(_DWORD *)(a1 + 264);
      v25 = v23;
      if ( !v24 )
      {
        ++*(_QWORD *)(a1 + 240);
        goto LABEL_50;
      }
      v26 = 1;
      v27 = *(_QWORD *)(a1 + 248);
      v28 = 0;
      v29 = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
      v30 = (v24 - 1) & v29;
      v31 = (_QWORD *)(v27 + 16LL * v30);
      v32 = *v31;
      if ( v25 != *v31 )
      {
        while ( v32 != -4096 )
        {
          if ( v32 == -8192 && !v28 )
            v28 = v31;
          v30 = (v24 - 1) & (v26 + v30);
          v31 = (_QWORD *)(v27 + 16LL * v30);
          v32 = *v31;
          if ( v25 == *v31 )
            goto LABEL_15;
          ++v26;
        }
        if ( !v28 )
          v28 = v31;
        v33 = *(_DWORD *)(a1 + 256);
        ++*(_QWORD *)(a1 + 240);
        v34 = v33 + 1;
        if ( 4 * v34 < 3 * v24 )
        {
          if ( v24 - *(_DWORD *)(a1 + 260) - v34 <= v24 >> 3 )
          {
            v59 = v29;
            sub_354C5D0(a1 + 240, v24);
            v48 = *(_DWORD *)(a1 + 264);
            if ( !v48 )
            {
LABEL_70:
              ++*(_DWORD *)(a1 + 256);
              BUG();
            }
            v49 = v48 - 1;
            v50 = 1;
            v47 = 0;
            v51 = *(_QWORD *)(a1 + 248);
            LODWORD(v52) = v49 & v59;
            v34 = *(_DWORD *)(a1 + 256) + 1;
            v28 = (__int64 *)(v51 + 16LL * (v49 & v59));
            v53 = *v28;
            if ( v25 != *v28 )
            {
              while ( v53 != -4096 )
              {
                if ( !v47 && v53 == -8192 )
                  v47 = v28;
                v52 = v49 & (unsigned int)(v52 + v50);
                v28 = (__int64 *)(v51 + 16 * v52);
                v53 = *v28;
                if ( v25 == *v28 )
                  goto LABEL_33;
                ++v50;
              }
              goto LABEL_54;
            }
          }
          goto LABEL_33;
        }
LABEL_50:
        sub_354C5D0(a1 + 240, 2 * v24);
        v41 = *(_DWORD *)(a1 + 264);
        if ( !v41 )
          goto LABEL_70;
        v42 = v41 - 1;
        v43 = *(_QWORD *)(a1 + 248);
        LODWORD(v44) = v42 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v34 = *(_DWORD *)(a1 + 256) + 1;
        v28 = (__int64 *)(v43 + 16LL * (unsigned int)v44);
        v45 = *v28;
        if ( v25 != *v28 )
        {
          v46 = 1;
          v47 = 0;
          while ( v45 != -4096 )
          {
            if ( v45 == -8192 && !v47 )
              v47 = v28;
            v44 = v42 & (unsigned int)(v44 + v46);
            v28 = (__int64 *)(v43 + 16 * v44);
            v45 = *v28;
            if ( v25 == *v28 )
              goto LABEL_33;
            ++v46;
          }
LABEL_54:
          if ( v47 )
            v28 = v47;
        }
LABEL_33:
        *(_DWORD *)(a1 + 256) = v34;
        if ( *v28 != -4096 )
          --*(_DWORD *)(a1 + 260);
        *v28 = v25;
        result = v28 + 1;
        *((_DWORD *)v28 + 2) = 0;
        goto LABEL_16;
      }
LABEL_15:
      result = v31 + 1;
LABEL_16:
      *result = v19;
      if ( !v6 )
        BUG();
      if ( (*(_BYTE *)v6 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
          v6 = *(_QWORD *)(v6 + 8);
      }
      v6 = *(_QWORD *)(v6 + 8);
      if ( v55 == (_DWORD *)v6 )
        return result;
    }
    v56 = v6;
    v18 = *(_QWORD **)(v14 + 120);
    v19 = 0x7FFFFFFF;
    do
    {
      if ( (*v18 & 6) == 0 )
      {
        v58 = *(_QWORD *)(*v18 & 0xFFFFFFFFFFFFFFF8LL);
        v20 = sub_35E8960(a1, v58);
        v21 = sub_35E86F0(a1, v58);
        if ( !sub_35E7250(a1, v21, a2) && v19 > v20 )
          v19 = v20;
      }
      v18 += 2;
    }
    while ( v17 != v18 );
    v6 = v56;
    v22 = sub_35E72A0(a1, v56);
    if ( v22 )
    {
      v36 = sub_2EBEE10(*(_QWORD *)(a1 + 64), v22);
      if ( *(_QWORD *)(a1 + 24) == *(_QWORD *)(v36 + 24) )
      {
LABEL_39:
        v37 = sub_35E8960(a1, v36);
        v38 = sub_35E86F0(a1, v36);
        if ( !sub_35E7250(a1, v38, a2) && v19 > v37 )
          v19 = v37;
      }
    }
    if ( v19 != 0x7FFFFFFF )
      goto LABEL_13;
    goto LABEL_37;
  }
  return result;
}
