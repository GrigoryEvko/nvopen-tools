// Function: sub_1A0CF30
// Address: 0x1a0cf30
//
void __fastcall sub_1A0CF30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rdx
  _QWORD *v8; // rbx
  _QWORD *v9; // r14
  unsigned __int64 v10; // r15
  _QWORD *v11; // rax
  int v12; // ecx
  int v13; // eax
  int v14; // esi
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rdi
  int v19; // eax
  __int64 v20; // rdi
  int v21; // ecx
  unsigned int v22; // eax
  __int64 *v23; // rdx
  __int64 v24; // rsi
  _BYTE *v25; // rax
  __int64 v26; // r12
  _QWORD *v27; // r15
  unsigned __int8 v28; // bl
  __int64 i; // rdi
  _QWORD *v30; // r13
  _QWORD *v31; // rax
  char v32; // dl
  __int64 v33; // rdi
  __int64 v34; // rax
  unsigned int v35; // esi
  _QWORD *v36; // rdx
  _QWORD *v37; // r9
  unsigned __int64 v38; // rdi
  _QWORD *v39; // rdi
  int v40; // edx
  int v41; // r10d
  int v42; // edx
  int v43; // r8d
  int v44; // eax
  int v45; // r8d
  __int64 v46; // [rsp+10h] [rbp-150h]
  __int64 v48; // [rsp+28h] [rbp-138h] BYREF
  __int64 v49[3]; // [rsp+30h] [rbp-130h] BYREF
  _QWORD *v50; // [rsp+48h] [rbp-118h]
  __int64 v51[4]; // [rsp+50h] [rbp-110h] BYREF
  _BYTE *v52; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v53; // [rsp+78h] [rbp-E8h]
  _BYTE v54[64]; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v55; // [rsp+C0h] [rbp-A0h] BYREF
  _BYTE *v56; // [rsp+C8h] [rbp-98h]
  _BYTE *v57; // [rsp+D0h] [rbp-90h]
  __int64 v58; // [rsp+D8h] [rbp-88h]
  int v59; // [rsp+E0h] [rbp-80h]
  _BYTE v60[120]; // [rsp+E8h] [rbp-78h] BYREF

  v7 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v8 = *(_QWORD **)(a2 - 8);
    v9 = &v8[(unsigned __int64)v7 / 8];
  }
  else
  {
    v9 = (_QWORD *)a2;
    v8 = (_QWORD *)(a2 - v7);
  }
  v53 = 0x800000000LL;
  v10 = 0xAAAAAAAAAAAAAAABLL * (v7 >> 3);
  v11 = v54;
  v52 = v54;
  v12 = 0;
  if ( (unsigned __int64)v7 > 0xC0 )
  {
    sub_16CD150((__int64)&v52, v54, 0xAAAAAAAAAAAAAAABLL * (v7 >> 3), 8, a5, a6);
    v12 = v53;
    v11 = &v52[8 * (unsigned int)v53];
  }
  if ( v8 != v9 )
  {
    do
    {
      if ( v11 )
        *v11 = *v8;
      v8 += 3;
      ++v11;
    }
    while ( v8 != v9 );
    v12 = v53;
  }
  LODWORD(v53) = v12 + v10;
  v13 = *(_DWORD *)(a1 + 56);
  if ( v13 )
  {
    v14 = v13 - 1;
    v15 = *(_QWORD *)(a1 + 40);
    v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = (__int64 *)(v15 + 16LL * v16);
    v18 = *v17;
    if ( a2 == *v17 )
    {
LABEL_12:
      *v17 = -16;
      --*(_DWORD *)(a1 + 48);
      ++*(_DWORD *)(a1 + 52);
    }
    else
    {
      v44 = 1;
      while ( v18 != -8 )
      {
        v45 = v44 + 1;
        v16 = v14 & (v44 + v16);
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( a2 == *v17 )
          goto LABEL_12;
        v44 = v45;
      }
    }
  }
  v48 = a2;
  v19 = *(_DWORD *)(a1 + 88);
  if ( v19 )
  {
    v20 = *(_QWORD *)(a1 + 72);
    v21 = v19 - 1;
    v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v23 = (__int64 *)(v20 + 8LL * v22);
    v24 = *v23;
    if ( a2 == *v23 )
    {
LABEL_15:
      *v23 = -16;
      --*(_DWORD *)(a1 + 80);
      ++*(_DWORD *)(a1 + 84);
      sub_1A02020(v49, (_QWORD *)(a1 + 96), &v48);
      v55 = v49[0];
      v25 = (_BYTE *)*v50;
      v58 = (__int64)v50;
      v56 = v25;
      v57 = v25 + 512;
      sub_1A0CA40(v51, (_QWORD *)(a1 + 96), &v55);
    }
    else
    {
      v42 = 1;
      while ( v24 != -8 )
      {
        v43 = v42 + 1;
        v22 = v21 & (v42 + v22);
        v23 = (__int64 *)(v20 + 8LL * v22);
        v24 = *v23;
        if ( a2 == *v23 )
          goto LABEL_15;
        v42 = v43;
      }
    }
  }
  sub_15F20C0((_QWORD *)a2);
  v59 = 0;
  v56 = v60;
  v57 = v60;
  v55 = 0;
  v58 = 8;
  if ( (_DWORD)v53 )
  {
    v26 = 0;
    v46 = 8LL * (unsigned int)v53;
    do
    {
      v27 = *(_QWORD **)&v52[v26];
      v28 = *((_BYTE *)v27 + 16);
      if ( v28 > 0x17u )
      {
        for ( i = v27[1]; i; i = v27[1] )
        {
          v30 = *(_QWORD **)(i + 8);
          if ( v30 || v28 != *((_BYTE *)sub_1648700(i) + 16) )
            break;
          v31 = v56;
          if ( v57 != v56 )
            goto LABEL_23;
          v39 = &v56[8 * HIDWORD(v58)];
          if ( v56 != (_BYTE *)v39 )
          {
            while ( (_QWORD *)*v31 != v27 )
            {
              if ( *v31 == -2 )
                v30 = v31;
              if ( v39 == ++v31 )
              {
                if ( !v30 )
                  goto LABEL_44;
                *v30 = v27;
                --v59;
                ++v55;
                goto LABEL_24;
              }
            }
            break;
          }
LABEL_44:
          if ( HIDWORD(v58) < (unsigned int)v58 )
          {
            ++HIDWORD(v58);
            *v39 = v27;
            ++v55;
          }
          else
          {
LABEL_23:
            sub_16CCBA0((__int64)&v55, (__int64)v27);
            if ( !v32 )
              break;
          }
LABEL_24:
          v27 = sub_1648700(v27[1]);
        }
        v33 = *(_QWORD *)(a1 + 40);
        v34 = *(unsigned int *)(a1 + 56);
        if ( (_DWORD)v34 )
        {
          v35 = (v34 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v36 = (_QWORD *)(v33 + 16LL * v35);
          v37 = (_QWORD *)*v36;
          if ( (_QWORD *)*v36 == v27 )
          {
LABEL_27:
            if ( v36 != (_QWORD *)(v33 + 16 * v34) )
            {
              v51[0] = (__int64)v27;
              sub_1A062A0(a1 + 64, v51);
            }
          }
          else
          {
            v40 = 1;
            while ( v37 != (_QWORD *)-8LL )
            {
              v41 = v40 + 1;
              v35 = (v34 - 1) & (v40 + v35);
              v36 = (_QWORD *)(v33 + 16LL * v35);
              v37 = (_QWORD *)*v36;
              if ( (_QWORD *)*v36 == v27 )
                goto LABEL_27;
              v40 = v41;
            }
          }
        }
      }
      v26 += 8;
    }
    while ( v26 != v46 );
    v38 = (unsigned __int64)v57;
    *(_BYTE *)(a1 + 752) = 1;
    if ( (_BYTE *)v38 != v56 )
      _libc_free(v38);
  }
  else
  {
    *(_BYTE *)(a1 + 752) = 1;
  }
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
}
