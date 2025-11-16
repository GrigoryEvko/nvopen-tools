// Function: sub_3554FD0
// Address: 0x3554fd0
//
__int64 __fastcall sub_3554FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  bool v6; // zf
  __int64 v9; // rcx
  __int64 v11; // rax
  __int64 v13; // r14
  unsigned int v14; // edx
  __int64 v15; // rsi
  unsigned int v16; // r11d
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // edx
  __int64 *v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 *v25; // r9
  int v26; // edx
  unsigned int v27; // eax
  __int64 *v28; // rdi
  __int64 v29; // r8
  __int64 v30; // rax
  _QWORD *v31; // r9
  _QWORD *v32; // r15
  unsigned __int8 v33; // r11
  char v34; // al
  __int64 v35; // rax
  _QWORD *v36; // r9
  unsigned __int8 v37; // r11
  __int64 v38; // r15
  __int64 v39; // r10
  char v40; // al
  char v41; // al
  unsigned int v42; // r11d
  int v43; // esi
  int v44; // r11d
  __int64 v45; // rax
  int v46; // edi
  int v47; // r10d
  __int64 v48; // [rsp+0h] [rbp-80h]
  unsigned __int8 v49; // [rsp+0h] [rbp-80h]
  _QWORD *v50; // [rsp+8h] [rbp-78h]
  unsigned __int8 v51; // [rsp+8h] [rbp-78h]
  __int64 v52; // [rsp+8h] [rbp-78h]
  char v53; // [rsp+8h] [rbp-78h]
  char v55; // [rsp+10h] [rbp-70h]
  _QWORD *v56; // [rsp+10h] [rbp-70h]
  unsigned __int8 v57; // [rsp+10h] [rbp-70h]
  _QWORD *v58; // [rsp+10h] [rbp-70h]
  _QWORD *v59; // [rsp+10h] [rbp-70h]
  __int64 v60; // [rsp+18h] [rbp-68h] BYREF
  char v61[32]; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 v62; // [rsp+40h] [rbp-40h]

  v6 = *(_DWORD *)(a1 + 200) == -1;
  v60 = a1;
  if ( v6 )
    return 0;
  v9 = *(_QWORD *)(a4 + 8);
  v11 = *(unsigned int *)(a4 + 24);
  v13 = a5;
  if ( (_DWORD)v11 )
  {
    v14 = (v11 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    a5 = v9 + 8LL * v14;
    v15 = *(_QWORD *)a5;
    if ( a1 == *(_QWORD *)a5 )
    {
LABEL_4:
      if ( a5 != v9 + 8 * v11 )
        return 0;
    }
    else
    {
      a5 = 1;
      while ( v15 != -4096 )
      {
        v42 = a5 + 1;
        v14 = (v11 - 1) & (a5 + v14);
        a5 = v9 + 8LL * v14;
        v15 = *(_QWORD *)a5;
        if ( a1 == *(_QWORD *)a5 )
          goto LABEL_4;
        a5 = v42;
      }
    }
  }
  v18 = *(unsigned int *)(a3 + 24);
  v19 = *(_QWORD *)(a3 + 8);
  if ( (_DWORD)v18 )
  {
    a5 = (unsigned int)(v18 - 1);
    v20 = a5 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v21 = (__int64 *)(v19 + 8LL * v20);
    v22 = *v21;
    if ( a1 == *v21 )
    {
LABEL_11:
      v16 = 1;
      if ( v21 != (__int64 *)(v19 + 8 * v18) )
        return v16;
    }
    else
    {
      v43 = 1;
      while ( v22 != -4096 )
      {
        v44 = v43 + 1;
        v20 = a5 & (v43 + v20);
        v21 = (__int64 *)(v19 + 8LL * v20);
        v22 = *v21;
        if ( a1 == *v21 )
          goto LABEL_11;
        v43 = v44;
      }
    }
  }
  sub_3546A90((__int64)v61, v13, (__int64 *)a1, v19, a5, (__int64)a6);
  v16 = v62;
  if ( v62 )
  {
    v30 = sub_3545E90(a6, v60);
    v31 = a6;
    v48 = *(_QWORD *)v30 + 32LL * *(unsigned int *)(v30 + 8);
    if ( *(_QWORD *)v30 == v48 )
    {
      v45 = sub_35459D0(a6, v60);
      v38 = *(_QWORD *)v45;
      v39 = *(_QWORD *)v45 + 32LL * *(unsigned int *)(v45 + 8);
      if ( v39 == *(_QWORD *)v45 )
        return 0;
      v36 = a6;
      v37 = 0;
    }
    else
    {
      v32 = *(_QWORD **)v30;
      v33 = 0;
      do
      {
        v50 = v31;
        v55 = v33;
        v34 = sub_3545640((__int64)v32, 0);
        v33 = v55;
        v31 = v50;
        if ( !v34 )
        {
          v53 = v55;
          v59 = v31;
          v41 = sub_3554FD0(*v32, a2, a3, a4, v13);
          v31 = v59;
          v33 = v41 | v53;
        }
        v32 += 4;
      }
      while ( (_QWORD *)v48 != v32 );
      v51 = v33;
      v56 = v31;
      v35 = sub_35459D0(v31, v60);
      v36 = v56;
      v37 = v51;
      v38 = *(_QWORD *)v35;
      v39 = *(_QWORD *)v35 + 32LL * *(unsigned int *)(v35 + 8);
      if ( v39 == *(_QWORD *)v35 )
      {
LABEL_24:
        if ( v37 )
        {
          v57 = v37;
          sub_3554C70(a2, &v60);
          return v57;
        }
        return 0;
      }
    }
    do
    {
      if ( ((*(__int64 *)(v38 + 8) >> 1) & 3) == 1 && !*(_DWORD *)(v38 + 24) )
      {
        v49 = v37;
        v52 = v39;
        v58 = v36;
        v40 = sub_3554FD0(*(_QWORD *)(v38 + 8) & 0xFFFFFFFFFFFFFFF8LL, a2, a3, a4, v13);
        v39 = v52;
        v36 = v58;
        v37 = v40 | v49;
      }
      v38 += 32;
    }
    while ( v38 != v39 );
    goto LABEL_24;
  }
  v23 = *(_QWORD *)(a2 + 8);
  v24 = *(unsigned int *)(a2 + 24);
  v25 = (__int64 *)(v23 + 8 * v24);
  if ( (_DWORD)v24 )
  {
    v26 = v24 - 1;
    v27 = v26 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
    v28 = (__int64 *)(v23 + 8LL * v27);
    v29 = *v28;
    if ( v60 == *v28 )
    {
LABEL_15:
      LOBYTE(v16) = v25 != v28;
    }
    else
    {
      v46 = 1;
      while ( v29 != -4096 )
      {
        v47 = v46 + 1;
        v27 = v26 & (v46 + v27);
        v28 = (__int64 *)(v23 + 8LL * v27);
        v29 = *v28;
        if ( v60 == *v28 )
          goto LABEL_15;
        v46 = v47;
      }
    }
  }
  return v16;
}
