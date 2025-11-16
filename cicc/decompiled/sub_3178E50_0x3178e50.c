// Function: sub_3178E50
// Address: 0x3178e50
//
unsigned __int64 __fastcall sub_3178E50(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r8
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned int v7; // esi
  __int64 v8; // r12
  int v9; // r15d
  __int64 *v10; // r11
  unsigned int v11; // ebx
  unsigned int v12; // r10d
  __int64 *v13; // rdi
  __int64 v14; // r9
  int v16; // eax
  int v17; // eax
  __int64 v18; // r13
  __int64 v19; // r15
  __int64 v20; // r15
  int v21; // ecx
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v25; // rsi
  int v26; // eax
  __int64 **v27; // r11
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // r14
  unsigned __int64 v32; // rdx
  __int64 v33; // rbx
  unsigned __int8 **v34; // rdi
  int v35; // ecx
  unsigned __int8 **v36; // r9
  unsigned __int64 v37; // rcx
  __int64 v38; // rbx
  int v39; // edx
  int v40; // r12d
  int v41; // eax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  __int64 v44; // r12
  int v45; // ebx
  unsigned int v46; // r15d
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rcx
  unsigned int v50; // edx
  __int64 *v51; // rsi
  __int64 v52; // r10
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rax
  int v56; // edi
  int v57; // esi
  int v58; // r8d
  int v59; // eax
  int v60; // esi
  __int64 v61; // rdi
  unsigned int v62; // edx
  __int64 v63; // rcx
  int v64; // eax
  int v65; // ecx
  __int64 v66; // rdi
  __int64 *v67; // rsi
  unsigned int v68; // ebx
  __int64 v69; // rdx
  __int64 v70; // [rsp+0h] [rbp-B0h]
  __int64 v71; // [rsp+8h] [rbp-A8h]
  __int64 **v72; // [rsp+10h] [rbp-A0h]
  __int64 v74; // [rsp+28h] [rbp-88h]
  unsigned __int64 v75; // [rsp+38h] [rbp-78h]
  int v76; // [rsp+44h] [rbp-6Ch]
  __int64 v77; // [rsp+48h] [rbp-68h]
  unsigned __int8 **v78; // [rsp+50h] [rbp-60h] BYREF
  __int64 v79; // [rsp+58h] [rbp-58h]
  _BYTE v80[80]; // [rsp+60h] [rbp-50h] BYREF

  v2 = a1;
  v76 = 0;
  v75 = 0;
  v70 = a1 + 96;
LABEL_2:
  while ( 1 )
  {
    v3 = a2;
    v4 = *(unsigned int *)(a2 + 8);
    v5 = 8 * v4 - 8;
    if ( !(_DWORD)v4 )
      return v75;
    while ( 1 )
    {
      LODWORD(v4) = v4 - 1;
      v6 = *(_QWORD *)(*(_QWORD *)a2 + v5);
      *(_DWORD *)(a2 + 8) = v4;
      v7 = *(_DWORD *)(v2 + 120);
      if ( !v7 )
      {
        ++*(_QWORD *)(v2 + 96);
        v74 = v6;
        goto LABEL_66;
      }
      v8 = *(_QWORD *)(v2 + 104);
      v9 = 1;
      v10 = 0;
      v11 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
      v12 = (v7 - 1) & v11;
      v13 = (__int64 *)(v8 + 8LL * v12);
      v14 = *v13;
      if ( v6 != *v13 )
        break;
LABEL_5:
      v5 -= 8;
      if ( !(_DWORD)v4 )
        return v75;
    }
    while ( v14 != -4096 )
    {
      if ( v10 || v14 != -8192 )
        v13 = v10;
      v12 = (v7 - 1) & (v9 + v12);
      v14 = *(_QWORD *)(v8 + 8LL * v12);
      if ( v6 == v14 )
        goto LABEL_5;
      ++v9;
      v10 = v13;
      v13 = (__int64 *)(v8 + 8LL * v12);
    }
    v16 = *(_DWORD *)(v2 + 112);
    v74 = v6;
    if ( !v10 )
      v10 = v13;
    ++*(_QWORD *)(v2 + 96);
    v17 = v16 + 1;
    if ( 4 * v17 < 3 * v7 )
    {
      if ( v7 - (v17 + *(_DWORD *)(v2 + 116)) > v7 >> 3 )
        goto LABEL_16;
      sub_CF28B0(v70, v7);
      v64 = *(_DWORD *)(v2 + 120);
      if ( v64 )
      {
        v65 = v64 - 1;
        v66 = *(_QWORD *)(v2 + 104);
        v3 = 1;
        v67 = 0;
        v68 = (v64 - 1) & v11;
        v10 = (__int64 *)(v66 + 8LL * v68);
        v69 = *v10;
        v17 = *(_DWORD *)(v2 + 112) + 1;
        if ( v74 != *v10 )
        {
          while ( v69 != -4096 )
          {
            if ( !v67 && v69 == -8192 )
              v67 = v10;
            v14 = (unsigned int)(v3 + 1);
            v68 = v65 & (v3 + v68);
            v10 = (__int64 *)(v66 + 8LL * v68);
            v69 = *v10;
            if ( v74 == *v10 )
              goto LABEL_16;
            v3 = (unsigned int)v14;
          }
          if ( v67 )
            v10 = v67;
        }
        goto LABEL_16;
      }
LABEL_93:
      ++*(_DWORD *)(v2 + 112);
      BUG();
    }
LABEL_66:
    sub_CF28B0(v70, 2 * v7);
    v59 = *(_DWORD *)(v2 + 120);
    if ( !v59 )
      goto LABEL_93;
    v60 = v59 - 1;
    v61 = *(_QWORD *)(v2 + 104);
    v62 = (v59 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
    v10 = (__int64 *)(v61 + 8LL * v62);
    v63 = *v10;
    v17 = *(_DWORD *)(v2 + 112) + 1;
    if ( v74 != *v10 )
    {
      v14 = 1;
      v3 = 0;
      while ( v63 != -4096 )
      {
        if ( v63 == -8192 && !v3 )
          v3 = (__int64)v10;
        v62 = v60 & (v14 + v62);
        v10 = (__int64 *)(v61 + 8LL * v62);
        v63 = *v10;
        if ( v74 == *v10 )
          goto LABEL_16;
        v14 = (unsigned int)(v14 + 1);
      }
      if ( v3 )
        v10 = (__int64 *)v3;
    }
LABEL_16:
    *(_DWORD *)(v2 + 112) = v17;
    if ( *v10 != -4096 )
      --*(_DWORD *)(v2 + 116);
    *v10 = v74;
    v18 = *(_QWORD *)(v74 + 56);
    v19 = v74 + 48;
    if ( v74 + 48 != v18 )
    {
      v77 = v74 + 48;
      v20 = v2;
      while ( 1 )
      {
        while ( 1 )
        {
          v24 = v18 - 24;
          v25 = *(_QWORD *)(v20 + 72);
          if ( !v18 )
            v24 = 0;
          v26 = *(_DWORD *)(v20 + 88);
          if ( v26 )
            break;
LABEL_25:
          v27 = *(__int64 ***)(v20 + 48);
          v28 = 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF);
          if ( (*(_BYTE *)(v24 + 7) & 0x40) != 0 )
          {
            v29 = *(_QWORD *)(v24 - 8);
            v30 = v29 + v28;
          }
          else
          {
            v29 = v24 - v28;
            v30 = v24;
          }
          v31 = v30 - v29;
          v79 = 0x400000000LL;
          v32 = v31 >> 5;
          v78 = (unsigned __int8 **)v80;
          v33 = v31 >> 5;
          if ( (unsigned __int64)v31 > 0x80 )
          {
            v71 = v29;
            v72 = v27;
            sub_C8D5F0((__int64)&v78, v80, v32, 8u, v3, v14);
            v36 = v78;
            v35 = v79;
            v32 = v31 >> 5;
            v27 = v72;
            v29 = v71;
            v34 = &v78[(unsigned int)v79];
          }
          else
          {
            v34 = (unsigned __int8 **)v80;
            v35 = 0;
            v36 = (unsigned __int8 **)v80;
          }
          if ( v31 > 0 )
          {
            v37 = 0;
            do
            {
              v34[v37 / 8] = *(unsigned __int8 **)(v29 + 4 * v37);
              v37 += 8LL;
              --v33;
            }
            while ( v33 );
            v36 = v78;
            v35 = v79;
          }
          LODWORD(v79) = v32 + v35;
          v38 = sub_DFCEF0(v27, (unsigned __int8 *)v24, v36, (unsigned int)(v32 + v35), 2);
          v40 = v39;
          if ( v78 != (unsigned __int8 **)v80 )
            _libc_free((unsigned __int64)v78);
          v41 = 1;
          if ( v40 != 1 )
            v41 = v76;
          v76 = v41;
          v42 = v38 + v75;
          if ( __OFADD__(v38, v75) )
          {
            v42 = 0x8000000000000000LL;
            if ( v38 > 0 )
              v42 = 0x7FFFFFFFFFFFFFFFLL;
          }
          v75 = v42;
          v18 = *(_QWORD *)(v18 + 8);
          if ( v77 == v18 )
          {
LABEL_39:
            v2 = v20;
            v19 = v74 + 48;
            goto LABEL_40;
          }
        }
        v21 = v26 - 1;
        v22 = (v26 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v23 = *(_QWORD *)(v25 + 16LL * v22);
        if ( v24 != v23 )
        {
          v56 = 1;
          while ( v23 != -4096 )
          {
            v3 = (unsigned int)(v56 + 1);
            v22 = v21 & (v56 + v22);
            v23 = *(_QWORD *)(v25 + 16LL * v22);
            if ( v24 == v23 )
              goto LABEL_21;
            ++v56;
          }
          goto LABEL_25;
        }
LABEL_21:
        v18 = *(_QWORD *)(v18 + 8);
        if ( v77 == v18 )
          goto LABEL_39;
      }
    }
LABEL_40:
    v43 = *(_QWORD *)(v74 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v19 != v43 )
    {
      if ( !v43 )
        BUG();
      v44 = v43 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v43 - 24) - 30 <= 0xA )
      {
        v45 = sub_B46E30(v44);
        if ( v45 )
        {
          v46 = 0;
          while ( 1 )
          {
            v47 = sub_B46EC0(v44, v46);
            if ( (unsigned __int8)sub_2A64220(*(__int64 **)(v2 + 56), v47) )
            {
              v48 = *(unsigned int *)(v2 + 120);
              v49 = *(_QWORD *)(v2 + 104);
              if ( !(_DWORD)v48 )
                goto LABEL_50;
              v50 = (v48 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v51 = (__int64 *)(v49 + 8LL * v50);
              v52 = *v51;
              if ( v47 != *v51 )
              {
                v57 = 1;
                while ( v52 != -4096 )
                {
                  v58 = v57 + 1;
                  v50 = (v48 - 1) & (v57 + v50);
                  v51 = (__int64 *)(v49 + 8LL * v50);
                  v52 = *v51;
                  if ( v47 == *v51 )
                    goto LABEL_49;
                  v57 = v58;
                }
LABEL_50:
                if ( (unsigned __int8)sub_3175050(v2, v74, v47) )
                {
                  v55 = *(unsigned int *)(a2 + 8);
                  if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
                  {
                    sub_C8D5F0(a2, (const void *)(a2 + 16), v55 + 1, 8u, v53, v54);
                    v55 = *(unsigned int *)(a2 + 8);
                  }
                  *(_QWORD *)(*(_QWORD *)a2 + 8 * v55) = v47;
                  ++*(_DWORD *)(a2 + 8);
                }
                goto LABEL_45;
              }
LABEL_49:
              if ( v51 == (__int64 *)(v49 + 8 * v48) )
                goto LABEL_50;
            }
LABEL_45:
            if ( ++v46 == v45 )
              goto LABEL_2;
          }
        }
      }
    }
  }
}
