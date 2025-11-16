// Function: sub_2578B50
// Address: 0x2578b50
//
__int64 __fastcall sub_2578B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  __int64 v9; // rax
  int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rcx
  int v16; // edx
  unsigned int v17; // eax
  __int64 v18; // r8
  unsigned __int8 *v19; // r14
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  char v25; // al
  int v26; // edx
  __int64 j; // r14
  int v29; // r9d
  int v30; // r8d
  unsigned int v31; // eax
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  int v36; // r10d
  unsigned int i; // eax
  __int64 v38; // rdi
  unsigned int v39; // eax
  unsigned __int64 v40; // [rsp+0h] [rbp-F0h]
  __int64 v41; // [rsp+8h] [rbp-E8h]
  _QWORD *v42; // [rsp+10h] [rbp-E0h]
  __int64 v45; // [rsp+28h] [rbp-C8h]
  __int64 v46; // [rsp+28h] [rbp-C8h]
  __int64 v47; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v48; // [rsp+40h] [rbp-B0h] BYREF
  void *v49; // [rsp+48h] [rbp-A8h]
  __int64 v50; // [rsp+50h] [rbp-A0h]
  __int64 v51; // [rsp+58h] [rbp-98h]
  __int64 v52; // [rsp+60h] [rbp-90h]
  __int64 v53; // [rsp+68h] [rbp-88h]
  __int64 v54; // [rsp+70h] [rbp-80h]
  __int64 v55; // [rsp+78h] [rbp-78h]
  _BYTE v56[8]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v57; // [rsp+88h] [rbp-68h]
  unsigned int v58; // [rsp+98h] [rbp-58h]
  __int64 v59; // [rsp+A8h] [rbp-48h]
  __int64 v60; // [rsp+B0h] [rbp-40h]
  __int64 v61; // [rsp+B8h] [rbp-38h]

  v48 = 0;
  v8 = sub_2568740(a3, a4);
  v49 = 0;
  v50 = 0;
  v51 = 0;
  sub_C7D6A0(0, 0, 8);
  v9 = *(unsigned int *)(v8 + 24);
  LODWORD(v51) = v9;
  if ( (_DWORD)v9 )
  {
    v49 = (void *)sub_C7D670(8 * v9, 8);
    v50 = *(_QWORD *)(v8 + 16);
    memcpy(v49, *(const void **)(v8 + 8), 8LL * (unsigned int)v51);
  }
  else
  {
    v49 = 0;
    v50 = 0;
  }
  v52 = *(_QWORD *)(v8 + 32);
  v53 = *(_QWORD *)(v8 + 40);
  v54 = *(_QWORD *)(v8 + 48);
  v55 = *(_QWORD *)(v8 + 56);
  sub_254C700((__int64)v56, a3 + 200);
  if ( *(_DWORD *)(a5 + 40) )
  {
    v10 = 0;
    v42 = (_QWORD *)(a1 + 72);
    v11 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(*(_QWORD *)(a5 + 32) + 8 * v11);
        v13 = *(_QWORD *)(v12 + 24);
        if ( *(_BYTE *)v13 > 0x1Cu )
          break;
LABEL_16:
        v11 = (unsigned int)(v10 + 1);
        v10 = v11;
        if ( *(_DWORD *)(a5 + 40) <= (unsigned int)v11 )
          goto LABEL_17;
      }
      v14 = v13 & 0xFFFFFFFFFFFFFFFBLL;
      v15 = v13 | 4;
      if ( (_DWORD)v51 )
        break;
LABEL_22:
      v33 = v53;
      while ( v59 != v33 || v54 != v60 || v55 != v61 )
      {
        v33 = sub_3106C80(&v48);
        v53 = v33;
        if ( v13 == v33 )
          goto LABEL_8;
      }
      v11 = (unsigned int)(v10 + 1);
      v10 = v11;
      if ( *(_DWORD *)(a5 + 40) <= (unsigned int)v11 )
        goto LABEL_17;
    }
    v16 = v51 - 1;
    v17 = (v51 - 1) & (v15 ^ (v15 >> 9));
    v18 = *((_QWORD *)v49 + v17);
    if ( v15 != v18 )
    {
      v29 = 1;
      while ( v18 != -4 )
      {
        v17 = v16 & (v29 + v17);
        v18 = *((_QWORD *)v49 + v17);
        if ( v15 == v18 )
          goto LABEL_8;
        ++v29;
      }
      v30 = 1;
      v31 = v16 & (v14 ^ (v14 >> 9));
      v32 = *((_QWORD *)v49 + v31);
      if ( v14 != v32 )
      {
        while ( v32 != -4 )
        {
          v31 = v16 & (v30 + v31);
          v32 = *((_QWORD *)v49 + v31);
          if ( v14 == v32 )
            goto LABEL_8;
          ++v30;
        }
        goto LABEL_22;
      }
    }
LABEL_8:
    v19 = *(unsigned __int8 **)v12;
    v45 = *(_QWORD *)(a2 + 208);
    v20 = sub_25096F0(v42);
    v21 = 0;
    v22 = v20;
    if ( v20 )
    {
      v41 = v20;
      v23 = sub_2554D30(*(_QWORD *)(v45 + 240), v20, 0);
      v24 = *(_QWORD *)(v45 + 240);
      v21 = *(_QWORD *)v24;
      if ( *(_QWORD *)v24 )
      {
        if ( !*(_BYTE *)(v24 + 16) )
        {
          v46 = v23;
          v21 = sub_BC1CD0(*(_QWORD *)v24, &unk_4F86630, v41) + 8;
          v22 = v46;
          goto LABEL_12;
        }
        v34 = *(unsigned int *)(v21 + 88);
        v35 = *(_QWORD *)(v21 + 72);
        if ( !(_DWORD)v34 )
          goto LABEL_45;
        v36 = 1;
        v40 = (unsigned __int64)(((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4)) << 32;
        for ( i = (v34 - 1)
                & (((0xBF58476D1CE4E5B9LL * (v40 | ((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4))) >> 31)
                 ^ (484763065 * (v40 | ((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4)))); ; i = (v34 - 1) & v39 )
        {
          v38 = v35 + 24LL * i;
          if ( *(_UNKNOWN **)v38 == &unk_4F86630 && v41 == *(_QWORD *)(v38 + 8) )
            break;
          if ( *(_QWORD *)v38 == -4096 && *(_QWORD *)(v38 + 8) == -4096 )
            goto LABEL_45;
          v39 = v36 + i;
          ++v36;
        }
        if ( v38 == v35 + 24 * v34 )
        {
LABEL_45:
          v22 = v23;
          v21 = 0;
          goto LABEL_12;
        }
        v21 = *(_QWORD *)(*(_QWORD *)(v38 + 16) + 24LL);
        if ( v21 )
        {
          v21 += 8;
          v22 = v23;
          goto LABEL_12;
        }
      }
      v22 = v23;
    }
LABEL_12:
    v25 = sub_98ED60(v19, v21, v13, v22, 0);
    *(_BYTE *)(a6 + 8) |= v25;
    *(_BYTE *)(a6 + 9) |= v25;
    v26 = *(unsigned __int8 *)v13;
    if ( (unsigned int)(v26 - 67) <= 0xC || (_BYTE)v26 == 63 )
    {
      for ( j = *(_QWORD *)(v13 + 16); j; j = *(_QWORD *)(j + 8) )
      {
        v47 = j;
        sub_25789E0(a5, &v47);
      }
    }
    goto LABEL_16;
  }
LABEL_17:
  sub_C7D6A0(v57, 8LL * v58, 8);
  return sub_C7D6A0((__int64)v49, 8LL * (unsigned int)v51, 8);
}
