// Function: sub_2284050
// Address: 0x2284050
//
__int64 __fastcall sub_2284050(__int64 a1, _BYTE *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rax
  __int64 *v11; // rbx
  __int64 *v12; // r13
  __int64 v13; // r12
  __int64 *v14; // rdx
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 *v19; // r15
  __int64 v20; // r14
  __int64 v21; // rsi
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 **v24; // rax
  __int64 v25; // r12
  int v26; // r9d
  unsigned int i; // eax
  __int64 v28; // rsi
  unsigned int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rcx
  _QWORD *v38; // rbx
  __int64 v39; // r13
  _QWORD *v40; // rax
  __int64 v41; // rdi
  _QWORD *v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rcx
  int v50; // eax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  int v60; // r8d
  unsigned __int64 v63; // [rsp+20h] [rbp-190h]
  __int64 v65; // [rsp+50h] [rbp-160h]
  _QWORD *v67; // [rsp+60h] [rbp-150h]
  __int64 *v68; // [rsp+68h] [rbp-148h]
  __int64 **v70; // [rsp+78h] [rbp-138h]
  __int64 v71; // [rsp+88h] [rbp-128h] BYREF
  __int64 *v72; // [rsp+90h] [rbp-120h] BYREF
  __int64 v73; // [rsp+98h] [rbp-118h]
  _BYTE v74[32]; // [rsp+A0h] [rbp-110h] BYREF
  _BYTE v75[48]; // [rsp+C0h] [rbp-F0h] BYREF
  char v76[48]; // [rsp+F0h] [rbp-C0h] BYREF
  _QWORD v77[18]; // [rsp+120h] [rbp-90h] BYREF

  v6 = a1;
  v70 = (__int64 **)a3;
  v65 = *(_QWORD *)(sub_227ED20(a4, &qword_4FDADA8, a3, a5) + 8);
  v72 = (__int64 *)v74;
  v73 = 0x400000000LL;
  v10 = a3;
  v11 = (__int64 *)a3[1];
  v12 = &v11[*((unsigned int *)v10 + 4)];
  if ( v11 != v12 )
  {
    v13 = *v11;
    v14 = (__int64 *)v74;
    v15 = v11 + 1;
    v16 = 0;
    while ( 1 )
    {
      v14[v16] = v13;
      v16 = (unsigned int)(v73 + 1);
      LODWORD(v73) = v73 + 1;
      if ( v12 == v15 )
        break;
      v13 = *v15;
      if ( v16 + 1 > (unsigned __int64)HIDWORD(v73) )
      {
        sub_C8D5F0((__int64)&v72, v74, v16 + 1, 8u, v8, v9);
        v16 = (unsigned int)v73;
      }
      v14 = v72;
      ++v15;
    }
  }
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&qword_4F82400);
  v18 = (unsigned int)v73;
  v68 = &v72[(unsigned int)v73];
  if ( v68 != v72 )
  {
    v19 = v72;
    v18 = ((unsigned int)&unk_4FDADD0 >> 9) ^ ((unsigned int)&unk_4FDADD0 >> 4);
    while ( 1 )
    {
      v20 = *v19;
      v17 = *(unsigned int *)(a5 + 328);
      v21 = *(_QWORD *)(a5 + 312);
      if ( !(_DWORD)v17 )
        goto LABEL_49;
      v17 = (unsigned int)(v17 - 1);
      v18 = (unsigned int)v17 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v22 = (__int64 *)(v21 + 16 * v18);
      v23 = *v22;
      if ( v20 != *v22 )
        break;
LABEL_12:
      v24 = (__int64 **)v22[1];
LABEL_13:
      if ( v24 != v70 )
        goto LABEL_9;
      v25 = *(_QWORD *)(v20 + 8);
      if ( a2[9] )
      {
        v18 = *(unsigned int *)(v65 + 88);
        v17 = *(_QWORD *)(v65 + 72);
        if ( (_DWORD)v18 )
        {
          v26 = 1;
          v63 = (unsigned __int64)(((unsigned int)&unk_4FDADD0 >> 9) ^ ((unsigned int)&unk_4FDADD0 >> 4)) << 32;
          for ( i = (v18 - 1)
                  & (((0xBF58476D1CE4E5B9LL * (v63 | ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4))) >> 31)
                   ^ (484763065 * (v63 | ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)))); ; i = (v18 - 1) & v29 )
          {
            v28 = v17 + 24LL * i;
            if ( *(_UNKNOWN **)v28 == &unk_4FDADD0 && v25 == *(_QWORD *)(v28 + 8) )
              break;
            if ( *(_QWORD *)v28 == -4096 && *(_QWORD *)(v28 + 8) == -4096 )
              goto LABEL_23;
            v29 = v26 + i;
            ++v26;
          }
          if ( v28 != v17 + 24 * v18 && *(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL) )
            goto LABEL_9;
        }
      }
LABEL_23:
      v30 = sub_BC1CD0(v65, &qword_4F8A320, *(_QWORD *)(v20 + 8));
      v31 = *(_QWORD *)a2;
      v71 = *(_QWORD *)(v30 + 8);
      if ( (unsigned __int8)sub_BBBC50(&v71, v31, v25) )
      {
        (*(void (__fastcall **)(_BYTE *, _QWORD, __int64, __int64))(**(_QWORD **)a2 + 16LL))(
          v75,
          *(_QWORD *)a2,
          v25,
          v65);
        if ( a2[8] )
        {
          memset(v77, 0, 0x60u);
          v35 = 0;
          BYTE4(v77[3]) = 1;
          v77[1] = &v77[4];
          LODWORD(v77[2]) = 2;
          v77[7] = &v77[10];
          LODWORD(v77[8]) = 2;
          BYTE4(v77[9]) = 1;
        }
        else
        {
          sub_C8CD80((__int64)v77, (__int64)&v77[4], (__int64)v75, v32, v33, v34);
          sub_C8CD80((__int64)&v77[6], (__int64)&v77[10], (__int64)v76, v57, v58, v59);
        }
        sub_BBE020(v65, v25, (__int64)v77, v35);
        sub_227AD40((__int64)v77);
        if ( v71 )
        {
          v38 = *(_QWORD **)(v71 + 432);
          v67 = &v38[4 * *(unsigned int *)(v71 + 440)];
          if ( v38 != v67 )
          {
            v39 = *(_QWORD *)a2;
            do
            {
              v77[0] = 0;
              v40 = (_QWORD *)sub_22077B0(0x10u);
              if ( v40 )
              {
                v40[1] = v25;
                *v40 = &unk_49DB0A8;
              }
              v41 = v77[0];
              v77[0] = v40;
              if ( v41 )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v41 + 8LL))(v41);
              v42 = v38;
              v44 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v39 + 32LL))(v39);
              if ( (v38[3] & 2) == 0 )
                v42 = (_QWORD *)*v38;
              (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *, _BYTE *))(v38[3] & 0xFFFFFFFFFFFFFFF8LL))(
                v42,
                v44,
                v43,
                v77,
                v75);
              if ( v77[0] )
                (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v77[0] + 8LL))(v77[0]);
              v38 += 4;
            }
            while ( v67 != v38 );
          }
        }
        sub_BBADB0(a1, (__int64)v75, v36, v37);
        if ( (unsigned __int8)sub_B19060(a1 + 48, (__int64)&unk_4F86C48, v45, v46)
          || !(unsigned __int8)sub_B19060(a1, (__int64)&qword_4F82400, v47, v48)
          && !(unsigned __int8)sub_B19060(a1, (__int64)&unk_4F86C48, v51, v52)
          && !(unsigned __int8)sub_B19060(a1, (__int64)&qword_4F82400, v53, v54)
          && !(unsigned __int8)sub_B19060(a1, (__int64)&unk_4F82428, v55, v56) )
        {
          v70 = (__int64 **)sub_2284040(a5, v70, v20, a4, a6, v65);
        }
        ++v19;
        sub_227AD40((__int64)v75);
        if ( v68 == v19 )
        {
LABEL_41:
          v6 = a1;
          goto LABEL_42;
        }
      }
      else
      {
LABEL_9:
        if ( v68 == ++v19 )
          goto LABEL_41;
      }
    }
    v50 = 1;
    while ( v23 != -4096 )
    {
      v60 = v50 + 1;
      v18 = (unsigned int)v17 & (v50 + (_DWORD)v18);
      v22 = (__int64 *)(v21 + 16LL * (unsigned int)v18);
      v23 = *v22;
      if ( v20 == *v22 )
        goto LABEL_12;
      v50 = v60;
    }
LABEL_49:
    v24 = 0;
    goto LABEL_13;
  }
LABEL_42:
  if ( *(_DWORD *)(v6 + 68) != *(_DWORD *)(v6 + 72)
    || !(unsigned __int8)sub_B19060(v6, (__int64)&qword_4F82400, v18, v17) )
  {
    sub_AE6EC0(v6, (__int64)&unk_4F82420);
  }
  sub_227AC60(v6, (__int64)&qword_4FDADA8);
  sub_227AC60(v6, (__int64)&unk_4F86C48);
  if ( v72 != (__int64 *)v74 )
    _libc_free((unsigned __int64)v72);
  return v6;
}
