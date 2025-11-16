// Function: sub_2F558F0
// Address: 0x2f558f0
//
__int64 __fastcall sub_2F558F0(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned int v10; // ebx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned int v14; // r10d
  __int16 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // eax
  unsigned __int64 v22; // r8
  unsigned int v23; // r10d
  int v24; // edx
  __int64 v25; // rdx
  __int64 result; // rax
  unsigned int v27; // r15d
  __int16 *v28; // r14
  __int64 *v29; // rcx
  __int64 *v30; // r9
  __int64 v31; // r10
  __int64 v32; // rdx
  unsigned int v33; // esi
  unsigned int v34; // ecx
  __int64 v35; // rdi
  int v36; // edx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r11
  __int64 v40; // rsi
  __int64 *v41; // rdx
  __int64 *v42; // rdi
  __int64 v43; // rdx
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // r9
  _QWORD *v46; // r8
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned int v50; // ecx
  unsigned int v51; // r14d
  float v52; // xmm1_4
  __int64 v53; // r9
  __int64 v54; // rcx
  __int64 v55; // r11
  __int64 v56; // rdx
  int v57; // eax
  _QWORD *v58; // rax
  _QWORD *v59; // r8
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r11
  __int64 v63; // rcx
  __int64 v64; // rcx
  unsigned int v65; // esi
  unsigned __int64 v66; // r15
  void *v67; // rdi
  size_t v68; // rdx
  __int64 v69; // rax
  _QWORD *v70; // [rsp+0h] [rbp-E0h]
  unsigned int v71; // [rsp+Ch] [rbp-D4h]
  __int64 v72; // [rsp+18h] [rbp-C8h]
  _QWORD *v73; // [rsp+20h] [rbp-C0h]
  _QWORD *v74; // [rsp+20h] [rbp-C0h]
  signed __int64 v75; // [rsp+28h] [rbp-B8h]
  signed __int64 v76; // [rsp+30h] [rbp-B0h]
  unsigned int v78; // [rsp+40h] [rbp-A0h]
  __int64 *v79; // [rsp+40h] [rbp-A0h]
  unsigned int v80; // [rsp+40h] [rbp-A0h]
  char v81; // [rsp+40h] [rbp-A0h]
  __int64 *v82; // [rsp+40h] [rbp-A0h]
  _QWORD *v83; // [rsp+40h] [rbp-A0h]
  unsigned int v84; // [rsp+48h] [rbp-98h]
  unsigned int v85; // [rsp+4Ch] [rbp-94h]
  __int64 v86; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v87; // [rsp+58h] [rbp-88h] BYREF
  __int64 v88; // [rsp+60h] [rbp-80h]
  __int64 v89; // [rsp+68h] [rbp-78h] BYREF
  __int64 v90; // [rsp+70h] [rbp-70h]

  v7 = a1[124];
  v8 = *(_QWORD *)(v7 + 280);
  v9 = *(_QWORD *)(v7 + 200);
  v10 = *(_DWORD *)(v7 + 208) - 1;
  v76 = *(_QWORD *)(v8 + 8);
  if ( *(_BYTE *)(v8 + 32) )
    v76 = *(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v75 = *(_QWORD *)(v8 + 16);
  if ( *(_BYTE *)(v8 + 33) )
    v75 = *(_QWORD *)(v8 + 16) & 0xFFFFFFFFFFFFFFF8LL | 6;
  if ( v10 > *(_DWORD *)(a3 + 12) )
  {
    *(_DWORD *)(a3 + 8) = 0;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v10, 4u, a5, a6);
    v67 = *(void **)a3;
    v68 = 4LL * v10;
  }
  else
  {
    v11 = *(unsigned int *)(a3 + 8);
    v12 = v10;
    if ( v11 <= v10 )
      v12 = *(unsigned int *)(a3 + 8);
    if ( v12 )
    {
      memset(*(void **)a3, 0, 4 * v12);
      v11 = *(unsigned int *)(a3 + 8);
    }
    if ( v10 <= v11 )
      goto LABEL_11;
    v66 = v10 - v11;
    if ( !v66 )
      goto LABEL_11;
    v67 = (void *)(*(_QWORD *)a3 + 4 * v11);
    v68 = 4 * v66;
    if ( !(4 * v66) )
      goto LABEL_11;
  }
  memset(v67, 0, v68);
LABEL_11:
  *(_DWORD *)(a3 + 8) = v10;
  v13 = a1[1];
  v72 = 24LL * a2;
  v84 = (v76 >> 1) & 3;
  v14 = *(_DWORD *)(*(_QWORD *)(v13 + 8) + v72 + 16) & 0xFFF;
  v15 = (__int16 *)(*(_QWORD *)(v13 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v13 + 8) + v72 + 16) >> 12));
  v85 = (v75 >> 1) & 3;
  while ( v15 )
  {
    v78 = v14;
    v16 = sub_2E21610(a1[5], *(_QWORD *)(a1[124] + 40LL), v14);
    v21 = sub_2E1AC90(v16, 1u, v17, v18, v19, v20);
    v23 = v78;
    if ( v21 )
    {
      v37 = *(_QWORD *)(a1[5] + 48LL) + 216LL * v78;
      v38 = *(unsigned int *)(v37 + 200);
      v39 = v37 + 8;
      v87 = &v89;
      v40 = *(unsigned int *)(v37 + 204);
      v86 = v37 + 8;
      v88 = 0x400000000LL;
      if ( !(_DWORD)v38 )
      {
        if ( (_DWORD)v40 )
        {
          v41 = (__int64 *)(v37 + 16);
          do
          {
            if ( (*(_DWORD *)((*v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v41 >> 1) & 3) > (*(_DWORD *)((v76 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v84) )
              break;
            v38 = (unsigned int)(v38 + 1);
            v41 += 2;
          }
          while ( (_DWORD)v40 != (_DWORD)v38 );
        }
        v89 = v39;
        v42 = &v89;
        v43 = 1;
        v44 = v40 | (v38 << 32);
        LODWORD(v88) = 1;
        v90 = v44;
        v45 = HIDWORD(v44);
        goto LABEL_38;
      }
      v62 = v37 + 16;
      if ( (_DWORD)v40 )
      {
        v37 += 104;
        v63 = 0;
        do
        {
          if ( (*(_DWORD *)((*(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v37 >> 1) & 3) > (*(_DWORD *)((v76 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v84) )
            break;
          v63 = (unsigned int)(v63 + 1);
          v37 += 8;
        }
        while ( (_DWORD)v40 != (_DWORD)v63 );
      }
      else
      {
        v63 = 0;
      }
      v89 = v62;
      v64 = v40 | (v63 << 32);
      LODWORD(v88) = 1;
      v90 = v64;
      if ( HIDWORD(v64) < (unsigned int)v64 )
      {
        sub_2E1A640((__int64)&v86, v76, v37, v64, v22);
        v43 = (unsigned int)v88;
        v42 = v87;
        v23 = v78;
        if ( (_DWORD)v88 )
        {
          LODWORD(v45) = *((_DWORD *)v87 + 3);
          LODWORD(v44) = *((_DWORD *)v87 + 2);
LABEL_38:
          v46 = (_QWORD *)a3;
          for ( LODWORD(v47) = 0; (unsigned int)v44 > (unsigned int)v45; LODWORD(v47) = v51 )
          {
            v48 = (__int64)&v42[2 * v43 - 2];
            v49 = *(_QWORD *)(*(_QWORD *)v48 + 16LL * *(unsigned int *)(v48 + 12));
            v50 = *(_DWORD *)((v49 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v49 >> 1) & 3;
            if ( v50 >= (*(_DWORD *)((v75 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v85) )
              break;
            while ( 1 )
            {
              v51 = v47;
              v47 = (unsigned int)(v47 + 1);
              if ( v50 <= (*(_DWORD *)((*(_QWORD *)(v9 + 8 * v47) & 0xFFFFFFFFFFFFFFF8LL) + 24) | 3u) )
                break;
              if ( (_DWORD)v47 == v10 )
                goto LABEL_50;
            }
            if ( v51 == v10 )
              break;
            v52 = *(float *)(*(_QWORD *)(*(_QWORD *)v48 + 8LL * *(unsigned int *)(v48 + 12) + 128) + 116LL);
            while ( 1 )
            {
              *(float *)(*v46 + 4LL * v51) = fmaxf(v52, *(float *)(*v46 + 4LL * v51));
              v53 = v86;
              v54 = v51 + 1;
              v55 = *(_QWORD *)(v87[2 * (unsigned int)v88 - 2] + 16LL * HIDWORD(v87[2 * (unsigned int)v88 - 1]) + 8);
              if ( *(_DWORD *)((*(_QWORD *)(v9 + 8 * v54) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= (*(_DWORD *)((v55 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                          | (unsigned int)(v55 >> 1) & 3) )
                break;
              if ( (_DWORD)v54 == v10 )
              {
                v42 = v87;
                goto LABEL_50;
              }
              ++v51;
            }
            v56 = (__int64)&v87[2 * (unsigned int)v88 - 2];
            v57 = *(_DWORD *)(v56 + 12) + 1;
            *(_DWORD *)(v56 + 12) = v57;
            v43 = (unsigned int)v88;
            v42 = v87;
            if ( v57 == LODWORD(v87[2 * (unsigned int)v88 - 1]) )
            {
              v65 = *(_DWORD *)(v53 + 192);
              if ( v65 )
              {
                v70 = v46;
                v71 = v23;
                sub_F03D40((__int64 *)&v87, v65);
                v43 = (unsigned int)v88;
                v42 = v87;
                v23 = v71;
                v46 = v70;
              }
            }
            if ( !(_DWORD)v43 )
              break;
            LODWORD(v45) = *((_DWORD *)v42 + 3);
            LODWORD(v44) = *((_DWORD *)v42 + 2);
          }
        }
LABEL_50:
        if ( v42 != &v89 )
        {
          v80 = v23;
          _libc_free((unsigned __int64)v42);
          v23 = v80;
        }
      }
    }
    v24 = *v15++;
    v14 = v24 + v23;
    if ( !(_WORD)v24 )
      break;
  }
  v25 = a1[1];
  result = *(_DWORD *)(*(_QWORD *)(v25 + 8) + v72 + 16) >> 12;
  v27 = *(_DWORD *)(*(_QWORD *)(v25 + 8) + v72 + 16) & 0xFFF;
  v28 = (__int16 *)(*(_QWORD *)(v25 + 56) + 2 * result);
  do
  {
    if ( !v28 )
      break;
    v29 = *(__int64 **)(*(_QWORD *)(a1[4] + 424LL) + 8LL * v27);
    if ( !v29 )
    {
      v73 = (_QWORD *)a1[4];
      v81 = qword_501EA48[8];
      v58 = (_QWORD *)sub_22077B0(0x68u);
      v59 = v73;
      v60 = v27;
      v61 = (__int64)v58;
      if ( v58 )
      {
        *v58 = v58 + 2;
        v58[1] = 0x200000000LL;
        v58[8] = v58 + 10;
        v58[9] = 0x200000000LL;
        if ( v81 )
        {
          v74 = v58;
          v83 = v59;
          v69 = sub_22077B0(0x30u);
          v59 = v83;
          v61 = (__int64)v74;
          v60 = v27;
          if ( v69 )
          {
            *(_DWORD *)(v69 + 8) = 0;
            *(_QWORD *)(v69 + 16) = 0;
            *(_QWORD *)(v69 + 24) = v69 + 8;
            *(_QWORD *)(v69 + 32) = v69 + 8;
            *(_QWORD *)(v69 + 40) = 0;
          }
          v74[12] = v69;
        }
        else
        {
          v58[12] = 0;
        }
      }
      v82 = (__int64 *)v61;
      *(_QWORD *)(v59[53] + 8 * v60) = v61;
      sub_2E11710(v59, v61, v27);
      v29 = v82;
    }
    v79 = v29;
    v30 = (__int64 *)sub_2E09D00(v29, v76);
    result = *v79;
    v31 = *v79 + 24LL * *((unsigned int *)v79 + 2);
    if ( (__int64 *)v31 != v30 )
    {
      LODWORD(v32) = 0;
      do
      {
        v33 = *(_DWORD *)((*v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v30 >> 1) & 3;
        result = *(_DWORD *)((v75 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v85;
        if ( v33 >= (unsigned int)result )
          break;
        while ( 1 )
        {
          v34 = v32;
          v32 = (unsigned int)(v32 + 1);
          result = *(_DWORD *)((*(_QWORD *)(v9 + 8 * v32) & 0xFFFFFFFFFFFFFFF8LL) + 24) | 3u;
          if ( v33 <= (unsigned int)result )
            break;
          if ( (_DWORD)v32 == v10 )
            goto LABEL_28;
        }
        if ( v34 == v10 )
          break;
        v35 = v34;
        while ( 1 )
        {
          LODWORD(v32) = v34;
          *(_DWORD *)(*(_QWORD *)a3 + 4 * v35) = unk_44D0BE0;
          v35 = ++v34;
          result = *(_DWORD *)((v30[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v30[1] >> 1) & 3;
          if ( *(_DWORD *)((*(_QWORD *)(v9 + 8LL * v34) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= (unsigned int)result )
            break;
          if ( v34 == v10 )
            goto LABEL_28;
        }
        v30 += 3;
      }
      while ( v30 != (__int64 *)v31 );
    }
LABEL_28:
    v36 = *v28++;
    v27 += v36;
  }
  while ( (_WORD)v36 );
  return result;
}
