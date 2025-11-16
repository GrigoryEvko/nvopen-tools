// Function: sub_227E230
// Address: 0x227e230
//
__int64 __fastcall sub_227E230(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 *v6; // rbx
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r9d
  unsigned int i; // eax
  __int64 v19; // rsi
  unsigned int v20; // eax
  __int64 *v21; // rbx
  __int64 v22; // r8
  __int64 *v23; // r14
  char v24; // al
  __int64 v25; // r9
  __int64 v26; // r12
  char v27; // si
  __int64 v28; // rdi
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // r10
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // r12
  __int64 *v36; // rbx
  __int64 v37; // r14
  __int64 *v38; // rdx
  __int64 **v39; // rax
  _QWORD *v40; // rax
  __int64 *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  int v45; // r10d
  unsigned int j; // eax
  _QWORD *v47; // rsi
  unsigned int v48; // eax
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  int v53; // eax
  __int64 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rcx
  unsigned __int64 v60; // [rsp+8h] [rbp-148h]
  __int64 v61; // [rsp+10h] [rbp-140h]
  char v62; // [rsp+27h] [rbp-129h]
  __int64 v63; // [rsp+28h] [rbp-128h]
  __int64 *v66; // [rsp+40h] [rbp-110h]
  __int64 *v67; // [rsp+48h] [rbp-108h]
  __int64 v68; // [rsp+48h] [rbp-108h]
  __int64 v70; // [rsp+60h] [rbp-F0h]
  __int64 v72; // [rsp+70h] [rbp-E0h] BYREF
  char v73[8]; // [rsp+78h] [rbp-D8h] BYREF
  char v74[16]; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v75; // [rsp+90h] [rbp-C0h]
  _QWORD v76[20]; // [rsp+B0h] [rbp-A0h] BYREF

  if ( *(_DWORD *)(a3 + 72) == *(_DWORD *)(a3 + 68)
    && (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, a3, (__int64)a4) )
  {
    return 0;
  }
  v62 = sub_B19060(a3 + 48, (__int64)&unk_4FDADA8, a3, (__int64)a4);
  if ( v62
    || !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v4, v5)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4FDADA8, v10, v11)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v10, v56)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4FDADC8, v10, v57) )
  {
    v6 = *(__int64 **)(a2 + 8);
    v7 = &v6[*(unsigned int *)(a2 + 16)];
    while ( v7 != v6 )
    {
      v8 = *v6++;
      sub_BBE020(*a1, *(_QWORD *)(v8 + 8), a3, v5);
    }
    return 0;
  }
  v12 = *(unsigned int *)(a3 + 72);
  if ( *(_DWORD *)(a3 + 68) == (_DWORD)v12 )
  {
    v62 = sub_B19060(a3, (__int64)&unk_4F82400, v10, v12);
    if ( !v62 )
      v62 = sub_B19060(a3, (__int64)&unk_4F82420, v58, v59);
  }
  v13 = *(_QWORD *)(a2 + 8);
  v61 = v13 + 8LL * *(unsigned int *)(a2 + 16);
  if ( v61 != v13 )
  {
    v70 = *(_QWORD *)(a2 + 8);
    while ( 1 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)v70 + 8LL);
      memset(v76, 0, 0x68u);
      v63 = v14;
      v15 = *(unsigned int *)(*a1 + 88);
      v16 = *(_QWORD *)(*a1 + 72);
      if ( !(_DWORD)v15 )
        goto LABEL_16;
      v17 = 1;
      v60 = (unsigned __int64)(((unsigned int)&unk_4FDADB0 >> 9) ^ ((unsigned int)&unk_4FDADB0 >> 4)) << 32;
      for ( i = (v15 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v60 | ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4))) >> 31)
               ^ (484763065 * (v60 | ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; i = (v15 - 1) & v20 )
      {
        v19 = v16 + 24LL * i;
        if ( *(_UNKNOWN **)v19 == &unk_4FDADB0 && v63 == *(_QWORD *)(v19 + 8) )
          break;
        if ( *(_QWORD *)v19 == -4096 && *(_QWORD *)(v19 + 8) == -4096 )
          goto LABEL_16;
        v20 = v17 + i;
        ++v17;
      }
      if ( v19 != v16 + 24 * v15 )
      {
        v22 = *(_QWORD *)(*(_QWORD *)(v19 + 16) + 24LL);
        if ( v22 )
          break;
      }
LABEL_16:
      if ( !v62 )
      {
        sub_BBE020(*a1, v63, a3, v16);
        if ( LOBYTE(v76[12]) )
        {
LABEL_20:
          LOBYTE(v76[12]) = 0;
          sub_227AD40((__int64)v76);
        }
      }
LABEL_17:
      v70 += 8;
      if ( v61 == v70 )
        return 0;
    }
    v24 = *(_BYTE *)(v22 + 24) & 1;
    if ( *(_DWORD *)(v22 + 24) >> 1 )
    {
      if ( v24 )
      {
        v21 = (__int64 *)(v22 + 32);
        v23 = (__int64 *)(v22 + 64);
      }
      else
      {
        v21 = *(__int64 **)(v22 + 32);
        v22 = 16LL * *(unsigned int *)(v22 + 40);
        v23 = (__int64 *)((char *)v21 + v22);
        if ( v21 == (__int64 *)((char *)v21 + v22) )
          goto LABEL_16;
      }
      do
      {
        if ( *v21 != -8192 && *v21 != -4096 )
          break;
        v21 += 2;
      }
      while ( v23 != v21 );
    }
    else
    {
      if ( v24 )
      {
        v54 = v22 + 32;
        v55 = 32;
      }
      else
      {
        v54 = *(_QWORD *)(v22 + 32);
        v55 = 16LL * *(unsigned int *)(v22 + 40);
      }
      v21 = (__int64 *)(v55 + v54);
      v23 = v21;
    }
    if ( v21 == v23 )
      goto LABEL_16;
    while ( 1 )
    {
      v25 = *v21;
      v26 = *a4;
      v27 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v27 )
      {
        v28 = v26 + 16;
        v29 = 7;
      }
      else
      {
        v33 = *(unsigned int *)(v26 + 24);
        v28 = *(_QWORD *)(v26 + 16);
        if ( !(_DWORD)v33 )
          goto LABEL_84;
        v29 = v33 - 1;
      }
      v16 = v29 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v30 = v28 + 16 * v16;
      v31 = *(_QWORD *)v30;
      if ( v25 == *(_QWORD *)v30 )
        goto LABEL_34;
      v53 = 1;
      while ( v31 != -4096 )
      {
        v22 = (unsigned int)(v53 + 1);
        v16 = v29 & (unsigned int)(v53 + v16);
        v30 = v28 + 16LL * (unsigned int)v16;
        v31 = *(_QWORD *)v30;
        if ( v25 == *(_QWORD *)v30 )
          goto LABEL_34;
        v53 = v22;
      }
      if ( v27 )
      {
        v49 = 128;
        goto LABEL_85;
      }
      v33 = *(unsigned int *)(v26 + 24);
LABEL_84:
      v49 = 16 * v33;
LABEL_85:
      v30 = v28 + v49;
LABEL_34:
      v32 = 128;
      if ( !v27 )
        v32 = 16LL * *(unsigned int *)(v26 + 24);
      if ( v30 == v28 + v32 )
      {
        v42 = a4[1];
        v43 = *(unsigned int *)(v42 + 24);
        v44 = *(_QWORD *)(v42 + 8);
        if ( (_DWORD)v43 )
        {
          v45 = 1;
          for ( j = (v43 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                      | ((unsigned __int64)(((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v43 - 1) & v48 )
          {
            v47 = (_QWORD *)(v44 + 24LL * j);
            if ( v25 == *v47 && a2 == v47[1] )
              break;
            if ( *v47 == -4096 && v47[1] == -4096 )
              goto LABEL_91;
            v48 = v45 + j;
            ++v45;
          }
        }
        else
        {
LABEL_91:
          v47 = (_QWORD *)(v44 + 24 * v43);
        }
        v68 = *v21;
        v73[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v47[2] + 24LL) + 16LL))(
                   *(_QWORD *)(v47[2] + 24LL),
                   a2,
                   a3,
                   a4);
        v72 = v68;
        sub_BBCF50((__int64)v74, v26, &v72, v73);
        v30 = v75;
      }
      if ( !*(_BYTE *)(v30 + 8) )
      {
        v21 += 2;
        goto LABEL_39;
      }
      if ( !LOBYTE(v76[12]) )
      {
        sub_C8CD80((__int64)v76, (__int64)&v76[4], a3, v16, v22, v25);
        sub_C8CD80((__int64)&v76[6], (__int64)&v76[10], a3 + 48, v50, v51, v52);
        LOBYTE(v76[12]) = 1;
      }
      v34 = v21[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v21[1] & 4) != 0 )
      {
        v16 = *(_QWORD *)v34;
        v21 += 2;
        v35 = *(_QWORD *)v34 + 8LL * *(unsigned int *)(v34 + 8);
      }
      else
      {
        v16 = (__int64)(v21 + 1);
        v21 += 2;
        if ( !v34 )
          goto LABEL_39;
        v35 = (__int64)v21;
      }
      if ( v16 != v35 )
      {
        v67 = v23;
        v66 = v21;
        v36 = (__int64 *)v16;
        while ( 1 )
        {
          v37 = *v36;
          if ( BYTE4(v76[3]) )
          {
            v22 = v76[1];
            v38 = (__int64 *)(v76[1] + 8LL * HIDWORD(v76[2]));
            v39 = (__int64 **)v76[1];
            if ( (__int64 *)v76[1] != v38 )
            {
              while ( (__int64 *)v37 != *v39 )
              {
                if ( v38 == (__int64 *)++v39 )
                  goto LABEL_63;
              }
              --HIDWORD(v76[2]);
              v38 = *(__int64 **)(v76[1] + 8LL * HIDWORD(v76[2]));
              *v39 = v38;
              ++v76[0];
            }
          }
          else
          {
            v41 = sub_C8CA60((__int64)v76, v37);
            if ( v41 )
            {
              *v41 = -2;
              ++LODWORD(v76[3]);
              ++v76[0];
            }
          }
LABEL_63:
          if ( !BYTE4(v76[9]) )
            goto LABEL_70;
          v40 = (_QWORD *)v76[7];
          v38 = (__int64 *)(v76[7] + 8LL * HIDWORD(v76[8]));
          if ( (__int64 *)v76[7] == v38 )
          {
LABEL_72:
            if ( HIDWORD(v76[8]) >= LODWORD(v76[8]) )
            {
LABEL_70:
              ++v36;
              sub_C8CC70((__int64)&v76[6], v37, (__int64)v38, v16, v22, v25);
              if ( (__int64 *)v35 == v36 )
                goto LABEL_69;
            }
            else
            {
              ++v36;
              ++HIDWORD(v76[8]);
              *v38 = v37;
              ++v76[6];
              if ( (__int64 *)v35 == v36 )
                goto LABEL_69;
            }
          }
          else
          {
            while ( v37 != *v40 )
            {
              if ( v38 == ++v40 )
                goto LABEL_72;
            }
            if ( (__int64 *)v35 == ++v36 )
            {
LABEL_69:
              v23 = v67;
              v21 = v66;
              break;
            }
          }
        }
      }
LABEL_39:
      if ( v21 != v23 )
      {
        while ( *v21 == -8192 || *v21 == -4096 )
        {
          v21 += 2;
          if ( v23 == v21 )
            goto LABEL_43;
        }
        if ( v23 != v21 )
          continue;
      }
LABEL_43:
      if ( LOBYTE(v76[12]) )
      {
        sub_BBE020(*a1, v63, (__int64)v76, v16);
        if ( LOBYTE(v76[12]) )
          goto LABEL_20;
        goto LABEL_17;
      }
      goto LABEL_16;
    }
  }
  return 0;
}
