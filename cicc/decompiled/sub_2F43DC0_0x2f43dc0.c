// Function: sub_2F43DC0
// Address: 0x2f43dc0
//
void __fastcall sub_2F43DC0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v9; // rbx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rsi
  char v13; // dl
  __int64 v14; // rbx
  _BYTE *v15; // r12
  _BYTE *v16; // r13
  _BYTE *v17; // rbx
  _QWORD *v18; // rsi
  signed int v19; // r13d
  _DWORD *v20; // r12
  _QWORD *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rbx
  _DWORD *v25; // r14
  __int64 v26; // r12
  char *v27; // rax
  __int64 v28; // rdx
  unsigned __int16 *v29; // rdi
  char *v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // r9
  __int64 v35; // rdx
  char v36; // al
  __int64 v37; // rax
  _BYTE *v38; // r13
  unsigned int *v39; // r12
  __int64 v40; // rbx
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rcx
  __int64 v47; // rdx
  unsigned int v48; // eax
  unsigned int v49; // edx
  unsigned __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rdi
  __int64 v53; // rax
  int v54; // edx
  unsigned int *v55; // r14
  unsigned int *v56; // r15
  unsigned int v57; // ebx
  unsigned int v58; // edx
  unsigned int *v59; // r12
  __int128 v60; // [rsp-20h] [rbp-130h]
  __int128 v61; // [rsp-20h] [rbp-130h]
  __int128 v62; // [rsp-20h] [rbp-130h]
  _BYTE *v64; // [rsp+28h] [rbp-E8h]
  _BYTE *v65; // [rsp+30h] [rbp-E0h]
  __int64 v66; // [rsp+30h] [rbp-E0h]
  __int64 v67; // [rsp+38h] [rbp-D8h]
  unsigned int *v68; // [rsp+38h] [rbp-D8h]
  unsigned int v69; // [rsp+38h] [rbp-D8h]
  _QWORD v70[4]; // [rsp+80h] [rbp-90h] BYREF
  void *v71; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v72; // [rsp+A8h] [rbp-68h]
  _BYTE v73[96]; // [rsp+B0h] [rbp-60h] BYREF

  *(_DWORD *)(a1 + 1136) = 0;
  v6 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( !v6 )
    return;
  v9 = 0;
  do
  {
    v10 = (unsigned int)v9;
    v11 = *(_QWORD *)(a2 + 32) + 40 * v9;
    if ( *(_BYTE *)v11 )
      goto LABEL_9;
    a4 = *(unsigned __int8 *)(v11 + 4);
    v12 = *(unsigned int *)(v11 + 8);
    v13 = *(_BYTE *)(v11 + 3) & 0x10;
    if ( (a4 & 1) != 0 || (a4 &= 2u, (_DWORD)a4) )
    {
      if ( !v13 || (int)v12 >= 0 )
        goto LABEL_9;
    }
    else
    {
      if ( !v13 )
      {
        if ( (unsigned int)(v12 - 1) > 0x3FFFFFFE )
          goto LABEL_9;
LABEL_53:
        v46 = *(_QWORD *)(a1 + 16);
        v47 = *(_QWORD *)(v46 + 8);
        a4 = *(_QWORD *)(v46 + 56);
        v48 = *(_DWORD *)(v47 + 24 * v12 + 16);
        v49 = v48 & 0xFFF;
        v50 = a4 + 2LL * (v48 >> 12);
        do
        {
          if ( !v50 )
            break;
          a4 = *(_QWORD *)(a1 + 1112);
          v50 += 2LL;
          *(_DWORD *)(a4 + 4LL * v49) = *(_DWORD *)(a1 + 1104);
          v49 += *(__int16 *)(v50 - 2);
        }
        while ( *(_WORD *)(v50 - 2) );
        goto LABEL_9;
      }
      if ( (*(_DWORD *)v11 & 0xFFF00) != 0 && (unsigned int)(v12 - 1) <= 0x3FFFFFFE )
        goto LABEL_53;
      if ( (int)v12 >= 0 )
        goto LABEL_9;
    }
    if ( !*(_QWORD *)(a1 + 368)
      || (v34 = *(_QWORD *)(a1 + 16),
          v35 = *(_QWORD *)(a1 + 8),
          LODWORD(v71) = *(_DWORD *)(v11 + 8),
          v36 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, void **))(a1 + 376))(a1 + 352, v34, v35, &v71),
          v10 = (unsigned int)v9,
          v36) )
    {
      v37 = *(unsigned int *)(a1 + 1136);
      a4 = *(unsigned int *)(a1 + 1140);
      if ( v37 + 1 > a4 )
      {
        v69 = v10;
        sub_C8D5F0(a1 + 1128, (const void *)(a1 + 1144), v37 + 1, 4u, v10, a6);
        v37 = *(unsigned int *)(a1 + 1136);
        v10 = v69;
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 1128) + 4 * v37) = v10;
      ++*(_DWORD *)(a1 + 1136);
    }
LABEL_9:
    ++v9;
  }
  while ( v6 > (unsigned int)v9 );
  if ( *(_DWORD *)(a1 + 1136) > 1u )
  {
    v14 = (__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 288LL) - *(_QWORD *)(*(_QWORD *)(a1 + 16) + 280LL)) >> 3;
    v71 = v73;
    v72 = 0xC00000000LL;
    if ( (unsigned int)v14 > 0xCuLL )
    {
      sub_C8D5F0((__int64)&v71, v73, (unsigned int)v14, 4u, v10, a6);
      memset(v71, 0, 4LL * (unsigned int)v14);
      LODWORD(v72) = v14;
    }
    else
    {
      if ( (_DWORD)v14 && 4LL * (unsigned int)v14 )
        memset(v73, 0, 4LL * (unsigned int)v14);
      LODWORD(v72) = v14;
    }
    v15 = *(_BYTE **)(a2 + 32);
    v16 = &v15[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
    v65 = v16;
    if ( v15 != v16 )
    {
      while ( 1 )
      {
        v17 = v15;
        if ( sub_2DADC00(v15) )
          break;
        v15 += 40;
        if ( v16 == v15 )
          goto LABEL_46;
      }
      if ( v16 != v15 )
      {
        while ( 2 )
        {
          v18 = *(_QWORD **)(a1 + 16);
          v19 = *((_DWORD *)v17 + 2);
          v20 = v71;
          v21 = v18;
          if ( v19 >= 0 )
          {
            v22 = v18[35];
            v23 = (v18[36] - v22) >> 3;
            if ( (_DWORD)v23 )
            {
              v64 = v17;
              v67 = (unsigned int)v23;
              v24 = 0;
              v25 = v71;
              while ( 1 )
              {
                v26 = *(_QWORD *)(v22 + 8 * v24);
                v27 = sub_E922F0(v21, v19);
                v29 = (unsigned __int16 *)&v27[2 * v28];
                v30 = v27;
                if ( v27 != (char *)v29 )
                {
                  while ( 1 )
                  {
                    v31 = *(unsigned __int16 *)v30;
                    a4 = *(unsigned __int16 *)v30;
                    if ( v31 - 1 <= 0x3FFFFFFE )
                    {
                      v32 = v31 >> 3;
                      v10 = *(unsigned __int16 *)(*(_QWORD *)v26 + 22LL);
                      if ( (unsigned int)v32 < (unsigned int)v10 )
                      {
                        a4 &= 7u;
                        v33 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v26 + 8LL) + v32);
                        if ( _bittest(&v33, a4) )
                          break;
                      }
                    }
                    v30 += 2;
                    if ( v29 == (unsigned __int16 *)v30 )
                      goto LABEL_31;
                  }
                  ++v25[v24];
                }
LABEL_31:
                if ( v67 == ++v24 )
                  break;
                v21 = *(_QWORD **)(a1 + 16);
                v22 = v21[35];
              }
              v17 = v64;
            }
LABEL_41:
            v38 = v17 + 40;
            if ( v17 + 40 == v65 )
              goto LABEL_46;
            while ( 1 )
            {
              v17 = v38;
              if ( sub_2DADC00(v38) )
                break;
              v38 += 40;
              if ( v65 == v38 )
                goto LABEL_46;
            }
            if ( v65 == v38 )
              goto LABEL_46;
            continue;
          }
          break;
        }
        if ( *(_QWORD *)(a1 + 368) )
        {
          LODWORD(v70[0]) = *((_DWORD *)v17 + 2);
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, _QWORD, _QWORD *))(a1 + 376))(
                  a1 + 352,
                  v18,
                  *(_QWORD *)(a1 + 8),
                  v70) )
            goto LABEL_41;
          v18 = *(_QWORD **)(a1 + 16);
        }
        v51 = v18[35];
        v10 = (v18[36] - v51) >> 3;
        v52 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL) + 16LL * (v19 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (_DWORD)v10 )
        {
          v10 = (unsigned int)v10;
          v53 = 0;
          a6 = (__int64)v20;
          while ( 1 )
          {
            a4 = *(unsigned __int16 *)(**(_QWORD **)(v51 + 8 * v53) + 24LL);
            v54 = *(_DWORD *)(*(_QWORD *)(v52 + 8) + 4 * (a4 >> 5));
            if ( _bittest(&v54, a4) )
              ++v20[v53];
            if ( (unsigned int)v10 == ++v53 )
              break;
            v51 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 280LL);
          }
        }
        goto LABEL_41;
      }
    }
LABEL_46:
    v39 = *(unsigned int **)(a1 + 1128);
    v40 = *(unsigned int *)(a1 + 1136);
    v68 = &v39[v40];
    if ( v39 != &v39[v40] )
    {
      _BitScanReverse64(&v41, (v40 * 4) >> 2);
      *((_QWORD *)&v60 + 1) = a1;
      *(_QWORD *)&v60 = a2;
      sub_2F41E70(v39, &v39[v40], 2LL * (int)(63 - (v41 ^ 0x3F)), a4, v10, a6, v60, (__int64)&v71);
      if ( (unsigned __int64)v40 > 16 )
      {
        *((_QWORD *)&v62 + 1) = a1;
        *(_QWORD *)&v62 = a2;
        sub_2F42160(v39, v39 + 16, (__int64)&v71, v43, v44, v45, v62, (__int64)&v71);
        if ( &v39[v40] != v39 + 16 )
        {
          v66 = a1;
          v55 = v39 + 16;
          do
          {
            v56 = v55;
            v70[0] = a2;
            v70[1] = v66;
            v70[2] = &v71;
            v57 = *v55;
            while ( 1 )
            {
              v58 = *(v56 - 1);
              v59 = v56--;
              if ( !sub_2F41AD0(v70, v57, v58) )
                break;
              v56[1] = *v56;
            }
            *v59 = v57;
            ++v55;
          }
          while ( v68 != v55 );
        }
      }
      else
      {
        *((_QWORD *)&v61 + 1) = a1;
        *(_QWORD *)&v61 = a2;
        sub_2F42160(v39, v68, v42, v43, v44, v45, v61, (__int64)&v71);
      }
    }
    if ( v71 != v73 )
      _libc_free((unsigned __int64)v71);
  }
}
