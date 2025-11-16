// Function: sub_1C2AD00
// Address: 0x1c2ad00
//
void __fastcall sub_1C2AD00(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v7; // r13d
  unsigned int v8; // esi
  __int64 *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rsi
  int v12; // r10d
  __int64 v13; // rcx
  unsigned int v14; // r9d
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 *v17; // r8
  __int64 v18; // rbx
  __int64 v19; // r13
  int v20; // ecx
  __int64 v21; // r9
  unsigned int v22; // edi
  unsigned __int64 v23; // r8
  __int64 v24; // rdx
  int v26; // r11d
  int v27; // r14d
  unsigned int v29; // r15d
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // r11d
  unsigned int v34; // eax
  int v35; // edx
  unsigned int v36; // r9d
  unsigned int v37; // esi
  __int64 v38; // r10
  int v39; // ecx
  unsigned __int64 v40; // r8
  __int64 v41; // rdx
  unsigned __int64 v42; // r8
  int v45; // edx
  int v46; // edi
  int v47; // edi
  __int64 v48; // r10
  unsigned int v49; // r9d
  int v50; // eax
  __int64 *v51; // rdx
  __int64 v52; // r8
  int v53; // r9d
  int v54; // eax
  int v55; // edx
  __int64 v56; // rbx
  unsigned int v57; // r14d
  int v58; // r15d
  int v59; // eax
  int v60; // edi
  int v61; // edi
  __int64 v62; // r10
  int v63; // esi
  unsigned int v64; // r9d
  __int64 *v65; // rcx
  __int64 v66; // r8
  int v67; // esi
  _QWORD *v69; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(unsigned int *)(v2 + 48);
  if ( !(_DWORD)v3 )
    return;
  v4 = *(_QWORD *)(v2 + 32);
  v7 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v8 = (v3 - 1) & v7;
  v9 = (__int64 *)(v4 + 16LL * v8);
  v10 = *v9;
  if ( *v9 == a2 )
  {
LABEL_3:
    if ( v9 == (__int64 *)(v4 + 16 * v3) || !v9[1] )
      return;
    v11 = *(unsigned int *)(a1 + 136);
    if ( (_DWORD)v11 )
    {
      v12 = v11 - 1;
      v13 = *(_QWORD *)(a1 + 120);
      v14 = (v11 - 1) & v7;
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      v17 = v15;
      if ( *v15 == a2 )
      {
LABEL_7:
        v18 = v17[1];
        v69 = (_QWORD *)(a1 + 88);
        goto LABEL_8;
      }
      v56 = *v15;
      v57 = (v11 - 1) & v7;
      v58 = 1;
      v51 = 0;
      while ( v56 != -8 )
      {
        if ( v56 == -16 && !v51 )
          v51 = v17;
        v57 = v12 & (v58 + v57);
        v17 = (__int64 *)(v13 + 16LL * v57);
        v56 = *v17;
        if ( *v17 == a2 )
          goto LABEL_7;
        ++v58;
      }
      v59 = *(_DWORD *)(a1 + 128);
      if ( !v51 )
        v51 = v17;
      ++*(_QWORD *)(a1 + 112);
      v50 = v59 + 1;
      if ( 4 * v50 < (unsigned int)(3 * v11) )
      {
        if ( (int)v11 - *(_DWORD *)(a1 + 132) - v50 > (unsigned int)v11 >> 3 )
          goto LABEL_41;
        sub_1C29D90(a1 + 112, v11);
        v60 = *(_DWORD *)(a1 + 136);
        if ( v60 )
        {
          v61 = v60 - 1;
          v62 = *(_QWORD *)(a1 + 120);
          v63 = 1;
          v64 = v61 & v7;
          v65 = 0;
          v50 = *(_DWORD *)(a1 + 128) + 1;
          v51 = (__int64 *)(v62 + 16LL * (v61 & v7));
          v66 = *v51;
          if ( *v51 != a2 )
          {
            while ( v66 != -8 )
            {
              if ( v66 == -16 && !v65 )
                v65 = v51;
              v64 = v61 & (v63 + v64);
              v51 = (__int64 *)(v62 + 16LL * v64);
              v66 = *v51;
              if ( *v51 == a2 )
                goto LABEL_41;
              ++v63;
            }
LABEL_61:
            if ( v65 )
              v51 = v65;
            goto LABEL_41;
          }
          goto LABEL_41;
        }
        goto LABEL_81;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 112);
    }
    sub_1C29D90(a1 + 112, 2 * v11);
    v46 = *(_DWORD *)(a1 + 136);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(a1 + 120);
      v49 = v47 & v7;
      v50 = *(_DWORD *)(a1 + 128) + 1;
      v51 = (__int64 *)(v48 + 16LL * (v47 & v7));
      v52 = *v51;
      if ( *v51 != a2 )
      {
        v67 = 1;
        v65 = 0;
        while ( v52 != -8 )
        {
          if ( v52 == -16 && !v65 )
            v65 = v51;
          v49 = v47 & (v67 + v49);
          v51 = (__int64 *)(v48 + 16LL * v49);
          v52 = *v51;
          if ( *v51 == a2 )
            goto LABEL_41;
          ++v67;
        }
        goto LABEL_61;
      }
LABEL_41:
      *(_DWORD *)(a1 + 128) = v50;
      if ( *v51 != -8 )
        --*(_DWORD *)(a1 + 132);
      v51[1] = 0;
      v18 = 0;
      *v51 = a2;
      v11 = *(unsigned int *)(a1 + 136);
      v69 = (_QWORD *)(a1 + 88);
      v13 = *(_QWORD *)(a1 + 120);
      if ( !(_DWORD)v11 )
        goto LABEL_44;
      v12 = v11 - 1;
      v14 = (v11 - 1) & v7;
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
LABEL_8:
      if ( v16 == a2 )
        goto LABEL_9;
      v54 = 1;
      while ( v16 != -8 )
      {
        v55 = v54 + 1;
        v14 = v12 & (v54 + v14);
        v15 = (__int64 *)(v13 + 16LL * v14);
        v16 = *v15;
        if ( *v15 == a2 )
          goto LABEL_9;
        v54 = v55;
      }
LABEL_44:
      v15 = (__int64 *)(v13 + 16 * v11);
LABEL_9:
      v19 = v15[1];
      v20 = *(_DWORD *)(v19 + 40);
      if ( !v20 )
        goto LABEL_15;
      v21 = *(_QWORD *)(v19 + 24);
      v22 = (unsigned int)(v20 - 1) >> 6;
      v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v20;
      v24 = 0;
      while ( 1 )
      {
        _RCX = *(_QWORD *)(v21 + 8 * v24);
        if ( v22 == (_DWORD)v24 )
          _RCX = v23 & *(_QWORD *)(v21 + 8 * v24);
        if ( _RCX )
          break;
        if ( v22 + 1 == ++v24 )
          goto LABEL_15;
      }
      __asm { tzcnt   rcx, rcx }
      v29 = ((_DWORD)v24 << 6) + _RCX;
      if ( v29 == -1 )
      {
LABEL_15:
        v26 = *(_DWORD *)(v18 + 8);
        v27 = *(_DWORD *)(v18 + 12);
      }
      else
      {
        do
        {
          v30 = *(_QWORD *)(*v69 + 8LL * v29);
          v31 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
          v32 = sub_3952EB0(v30, v31);
          v33 = v32;
          v34 = v29 + 1;
          v26 = *(_DWORD *)(v18 + 8) + v33;
          v27 = *(_DWORD *)(v18 + 12) + HIDWORD(v32);
          *(_DWORD *)(v18 + 8) = v26;
          *(_DWORD *)(v18 + 12) = v27;
          v35 = *(_DWORD *)(v19 + 40);
          if ( v35 == v29 + 1 )
            break;
          v36 = v34 >> 6;
          v37 = (unsigned int)(v35 - 1) >> 6;
          if ( v34 >> 6 > v37 )
            break;
          v38 = *(_QWORD *)(v19 + 24);
          v39 = 64 - (v34 & 0x3F);
          v40 = 0xFFFFFFFFFFFFFFFFLL >> v39;
          v41 = v36;
          if ( v39 == 64 )
            v40 = 0;
          v42 = ~v40;
          while ( 1 )
          {
            _RAX = *(_QWORD *)(v38 + 8 * v41);
            if ( v36 == (_DWORD)v41 )
              _RAX = v42 & *(_QWORD *)(v38 + 8 * v41);
            if ( (_DWORD)v41 == v37 )
              _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*(_DWORD *)(v19 + 40);
            if ( _RAX )
              break;
            if ( v37 < (unsigned int)++v41 )
              goto LABEL_16;
          }
          __asm { tzcnt   rax, rax }
          v29 = ((_DWORD)v41 << 6) + _RAX;
        }
        while ( v29 != -1 );
      }
LABEL_16:
      if ( *(_DWORD *)(a1 + 36) >= v27 )
        v27 = *(_DWORD *)(a1 + 36);
      if ( *(_DWORD *)(a1 + 32) >= v26 )
        v26 = *(_DWORD *)(a1 + 32);
      *(_DWORD *)(a1 + 36) = v27;
      *(_DWORD *)(a1 + 32) = v26;
      *(_QWORD *)v18 = *(_QWORD *)(v18 + 8);
      sub_1C29FB0(a1, a2);
      return;
    }
LABEL_81:
    ++*(_DWORD *)(a1 + 128);
    BUG();
  }
  v45 = 1;
  while ( v10 != -8 )
  {
    v53 = v45 + 1;
    v8 = (v3 - 1) & (v45 + v8);
    v9 = (__int64 *)(v4 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == a2 )
      goto LABEL_3;
    v45 = v53;
  }
}
