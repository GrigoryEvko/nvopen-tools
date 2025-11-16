// Function: sub_20EA7E0
// Address: 0x20ea7e0
//
void __fastcall sub_20EA7E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  _QWORD *v6; // rdi
  unsigned int v7; // eax
  char v8; // dl
  __int64 v9; // rax
  unsigned __int64 v10; // r11
  __int64 v11; // r15
  _QWORD *v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // rax
  __int64 v16; // r12
  int v17; // r10d
  unsigned __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // r9
  __int64 v21; // r11
  __int64 v22; // r15
  unsigned __int64 v23; // r14
  __int64 *v24; // rdx
  int v25; // r9d
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 *v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // r12
  __int64 v31; // r14
  __int64 *v32; // rax
  __int64 *v33; // rsi
  unsigned int v34; // edi
  __int64 *v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // r10
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r15
  __int64 v42; // r14
  __int64 *v43; // rcx
  int v44; // r9d
  __int64 v45; // r10
  __int64 v46; // rax
  _QWORD *v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  int v50; // eax
  unsigned int v51; // r15d
  __int64 v52; // rcx
  __int64 v53; // rax
  _QWORD *v54; // rsi
  _QWORD *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r14
  _QWORD *v58; // rdi
  int v59; // r15d
  __int64 v60; // r14
  __int64 *v61; // rax
  __int64 v62; // r11
  _QWORD *v63; // rdx
  __int64 v64; // [rsp+8h] [rbp-F8h]
  __int64 v65; // [rsp+10h] [rbp-F0h]
  int v66; // [rsp+18h] [rbp-E8h]
  __int64 v67; // [rsp+28h] [rbp-D8h]
  __int64 v68; // [rsp+28h] [rbp-D8h]
  __int64 v69; // [rsp+38h] [rbp-C8h] BYREF
  _QWORD *v70; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+48h] [rbp-B8h]
  _QWORD v72[22]; // [rsp+50h] [rbp-B0h] BYREF

  v6 = v72;
  v71 = 0x800000001LL;
  v7 = 1;
  v70 = v72;
  v72[0] = a2;
  v72[1] = a3;
  do
  {
    while ( 1 )
    {
      v29 = &v6[2 * v7 - 2];
      v30 = *v29;
      v31 = v29[1];
      LODWORD(v71) = v7 - 1;
      v32 = *(__int64 **)(a1 + 280);
      if ( *(__int64 **)(a1 + 288) != v32 )
        goto LABEL_2;
      v33 = &v32[*(unsigned int *)(a1 + 300)];
      v34 = *(_DWORD *)(a1 + 300);
      if ( v32 != v33 )
        break;
LABEL_64:
      if ( v34 < *(_DWORD *)(a1 + 296) )
      {
        *(_DWORD *)(a1 + 300) = v34 + 1;
        *v33 = v31;
        ++*(_QWORD *)(a1 + 272);
        goto LABEL_3;
      }
LABEL_2:
      sub_16CCBA0(a1 + 272, v31);
      if ( !v8 )
        goto LABEL_24;
LABEL_3:
      v9 = *(_QWORD *)(v31 + 8);
      v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v9 & 6) == 0 )
      {
LABEL_44:
        v38 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
        if ( v10 )
        {
          v39 = *(_QWORD *)(v10 + 16);
          if ( v39 )
          {
            v40 = *(_QWORD *)(v39 + 24);
            goto LABEL_47;
          }
        }
        v57 = *(unsigned int *)(v38 + 544);
        v58 = *(_QWORD **)(v38 + 536);
        v69 = v9;
        v59 = v57;
        v60 = (__int64)&v58[2 * v57];
        v61 = sub_20EA370(v58, v60, &v69);
        if ( (__int64 *)v60 == v61 )
        {
          if ( !v59 )
            goto LABEL_78;
        }
        else if ( (*(_DWORD *)((*v61 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v61 >> 1) & 3) <= *(_DWORD *)(v62 + 24) )
        {
LABEL_78:
          v40 = v61[1];
LABEL_47:
          v41 = *(_QWORD *)(v40 + 64);
          v67 = *(_QWORD *)(v40 + 72);
          if ( v67 != v41 )
          {
            while ( 1 )
            {
              v48 = *(_QWORD *)(*(_QWORD *)(v38 + 392) + 16LL * *(unsigned int *)(*(_QWORD *)v41 + 48LL) + 8);
              v49 = v48 & 0xFFFFFFFFFFFFFFF8LL;
              v50 = (v48 >> 1) & 3;
              if ( v50 )
                v42 = (2LL * (v50 - 1)) | v49;
              else
                v42 = *(_QWORD *)v49 & 0xFFFFFFFFFFFFFFF8LL | 6;
              v43 = (__int64 *)sub_1DB3C70((__int64 *)v30, v42);
              if ( v43 != (__int64 *)(*(_QWORD *)v30 + 24LL * *(unsigned int *)(v30 + 8))
                && (*(_DWORD *)((*v43 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v43 >> 1) & 3) <= (*(_DWORD *)((v42 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v42 >> 1) & 3) )
              {
                v45 = v43[2];
                if ( v45 )
                {
                  v46 = (unsigned int)v71;
                  if ( (unsigned int)v71 >= HIDWORD(v71) )
                  {
                    v65 = v43[2];
                    sub_16CD150((__int64)&v70, v72, 0, 16, a5, v44);
                    v46 = (unsigned int)v71;
                    v45 = v65;
                  }
                  v47 = &v70[2 * v46];
                  *v47 = v30;
                  v47[1] = v45;
                  LODWORD(v71) = v71 + 1;
                }
              }
              v41 += 8;
              if ( v67 == v41 )
                break;
              v38 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
            }
          }
          goto LABEL_24;
        }
        v61 -= 2;
        goto LABEL_78;
      }
LABEL_4:
      v11 = 0;
      if ( v10 )
        v11 = *(_QWORD *)(v10 + 16);
      v12 = *(_QWORD **)(a1 + 184);
      v13 = *(_QWORD **)(a1 + 176);
      if ( v12 == v13 )
      {
        v14 = &v13[*(unsigned int *)(a1 + 196)];
        if ( v13 == v14 )
        {
          v63 = *(_QWORD **)(a1 + 176);
        }
        else
        {
          do
          {
            if ( v11 == *v13 )
              break;
            ++v13;
          }
          while ( v14 != v13 );
          v63 = v14;
        }
      }
      else
      {
        v14 = &v12[*(unsigned int *)(a1 + 192)];
        v13 = sub_16CC9F0(a1 + 168, v11);
        if ( v11 == *v13 )
        {
          v36 = *(_QWORD *)(a1 + 184);
          if ( v36 == *(_QWORD *)(a1 + 176) )
            v37 = *(unsigned int *)(a1 + 196);
          else
            v37 = *(unsigned int *)(a1 + 192);
          v63 = (_QWORD *)(v36 + 8 * v37);
        }
        else
        {
          v15 = *(_QWORD *)(a1 + 184);
          if ( v15 != *(_QWORD *)(a1 + 176) )
          {
            v13 = (_QWORD *)(v15 + 8LL * *(unsigned int *)(a1 + 192));
            goto LABEL_10;
          }
          v13 = (_QWORD *)(v15 + 8LL * *(unsigned int *)(a1 + 196));
          v63 = v13;
        }
      }
      while ( v63 != v13 && *v13 >= 0xFFFFFFFFFFFFFFFELL )
        ++v13;
LABEL_10:
      if ( v13 == v14 )
        goto LABEL_24;
      v16 = *(_QWORD *)(a1 + 16);
      v17 = *(_DWORD *)(*(_QWORD *)(v11 + 32) + 48LL);
      v18 = *(unsigned int *)(v16 + 408);
      v19 = v17 & 0x7FFFFFFF;
      v20 = v17 & 0x7FFFFFFF;
      v21 = 8 * v20;
      if ( (v17 & 0x7FFFFFFFu) >= (unsigned int)v18 || (v22 = *(_QWORD *)(*(_QWORD *)(v16 + 400) + 8LL * v19)) == 0 )
      {
        v51 = v19 + 1;
        if ( (unsigned int)v18 < v19 + 1 )
        {
          v53 = v51;
          if ( v51 >= v18 )
          {
            if ( v51 > v18 )
            {
              if ( v51 > (unsigned __int64)*(unsigned int *)(v16 + 412) )
              {
                v64 = v17 & 0x7FFFFFFF;
                v66 = v17;
                sub_16CD150(v16 + 400, (const void *)(v16 + 416), v51, 8, a5, v20);
                v18 = *(unsigned int *)(v16 + 408);
                v20 = v64;
                v21 = 8 * v64;
                v17 = v66;
                v53 = v51;
              }
              v52 = *(_QWORD *)(v16 + 400);
              v54 = (_QWORD *)(v52 + 8 * v53);
              v55 = (_QWORD *)(v52 + 8 * v18);
              v56 = *(_QWORD *)(v16 + 416);
              if ( v54 != v55 )
              {
                do
                  *v55++ = v56;
                while ( v54 != v55 );
                v52 = *(_QWORD *)(v16 + 400);
              }
              *(_DWORD *)(v16 + 408) = v51;
              goto LABEL_62;
            }
          }
          else
          {
            *(_DWORD *)(v16 + 408) = v51;
          }
        }
        v52 = *(_QWORD *)(v16 + 400);
LABEL_62:
        v68 = v20;
        *(_QWORD *)(v52 + v21) = sub_1DBA290(v17);
        v22 = *(_QWORD *)(*(_QWORD *)(v16 + 400) + 8 * v68);
        sub_1DBB110((_QWORD *)v16, v22);
      }
      v23 = *(_QWORD *)(v31 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      v24 = (__int64 *)sub_1DB3C70((__int64 *)v22, v23 | 2);
      if ( v24 == (__int64 *)(*(_QWORD *)v22 + 24LL * *(unsigned int *)(v22 + 8))
        || (*(_DWORD *)((*v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v24 >> 1) & 3) > (*(_DWORD *)(v23 + 24)
                                                                                              | 1u) )
      {
        v26 = (unsigned int)v71;
        v27 = 0;
        if ( (unsigned int)v71 >= HIDWORD(v71) )
          goto LABEL_38;
      }
      else
      {
        v27 = v24[2];
        v26 = (unsigned int)v71;
        if ( (unsigned int)v71 >= HIDWORD(v71) )
        {
LABEL_38:
          sub_16CD150((__int64)&v70, v72, 0, 16, a5, v25);
          v26 = (unsigned int)v71;
        }
      }
      v28 = &v70[2 * v26];
      *v28 = v22;
      v6 = v70;
      v28[1] = v27;
      v7 = v71 + 1;
      LODWORD(v71) = v7;
      if ( !v7 )
        goto LABEL_25;
    }
    v35 = 0;
    while ( v31 != *v32 )
    {
      if ( *v32 == -2 )
        v35 = v32;
      if ( v33 == ++v32 )
      {
        if ( !v35 )
          goto LABEL_64;
        *v35 = v31;
        --*(_DWORD *)(a1 + 304);
        ++*(_QWORD *)(a1 + 272);
        v9 = *(_QWORD *)(v31 + 8);
        v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v9 & 6) != 0 )
          goto LABEL_4;
        goto LABEL_44;
      }
    }
LABEL_24:
    v7 = v71;
    v6 = v70;
  }
  while ( (_DWORD)v71 );
LABEL_25:
  if ( v6 != v72 )
    _libc_free((unsigned __int64)v6);
}
