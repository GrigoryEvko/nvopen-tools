// Function: sub_AEE3F0
// Address: 0xaee3f0
//
void __fastcall sub_AEE3F0(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdi
  int v5; // ecx
  unsigned int v6; // eax
  __int64 v7; // rdx
  int v8; // r8d
  _QWORD *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  int v12; // r10d
  _QWORD *v13; // r9
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  _BYTE *v16; // rax
  _BYTE *v17; // r13
  unsigned int v18; // edx
  __int64 v19; // rsi
  int v20; // eax
  unsigned __int8 v21; // al
  _BYTE **v22; // rbx
  __int64 v23; // rdx
  _BYTE **v24; // r14
  unsigned int v25; // ecx
  _BYTE *v26; // r8
  _BYTE *v27; // r12
  int v28; // r8d
  __int64 v29; // r9
  int v30; // r8d
  unsigned int v31; // ecx
  _BYTE *v32; // r10
  int v33; // r11d
  unsigned __int8 v34; // cl
  __int64 v35; // r8
  _BYTE *v36; // rcx
  __int64 v37; // rsi
  int v38; // r8d
  _QWORD *v39; // rdi
  unsigned int v40; // ebx
  __int64 v41; // rcx
  int v42; // r10d
  int v43; // r10d
  _QWORD *v44; // r8
  __int64 v45; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v46; // [rsp-E0h] [rbp-E0h]
  __int64 v47; // [rsp-D8h] [rbp-D8h]
  __int64 v48; // [rsp-D0h] [rbp-D0h]
  _QWORD *v49; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v50; // [rsp-C0h] [rbp-C0h]
  _QWORD v51[23]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( a2 )
  {
    v3 = *(_DWORD *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 8);
    if ( v3 )
    {
      v5 = v3 - 1;
      v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = *(_QWORD *)(v4 + 16LL * v6);
      if ( a2 == v7 )
        return;
      v8 = 1;
      while ( v7 != -4096 )
      {
        v6 = v5 & (v8 + v6);
        v7 = *(_QWORD *)(v4 + 16LL * v6);
        if ( a2 == v7 )
          return;
        ++v8;
      }
    }
    v9 = v51;
    v51[0] = a2;
    v50 = 0x1000000001LL;
    LODWORD(a2) = 0;
    v10 = 0;
    v11 = 1;
    v49 = v51;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    while ( 1 )
    {
      v17 = (_BYTE *)v9[v11 - 1];
      if ( !(_DWORD)a2 )
        break;
      v12 = 1;
      v13 = 0;
      v14 = (a2 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v15 = (_QWORD *)(v10 + 8LL * v14);
      v16 = (_BYTE *)*v15;
      if ( v17 == (_BYTE *)*v15 )
      {
LABEL_9:
        sub_AED7C0(a1, (__int64)v17);
        a2 = (unsigned int)v48;
        v10 = v46;
        v11 = (unsigned int)(v50 - 1);
        LODWORD(v50) = v50 - 1;
        goto LABEL_10;
      }
      while ( v16 != (_BYTE *)-4096LL )
      {
        if ( v13 || v16 != (_BYTE *)-8192LL )
          v15 = v13;
        v14 = (a2 - 1) & (v12 + v14);
        v16 = *(_BYTE **)(v10 + 8LL * v14);
        if ( v17 == v16 )
          goto LABEL_9;
        ++v12;
        v13 = v15;
        v15 = (_QWORD *)(v10 + 8LL * v14);
      }
      if ( !v13 )
        v13 = v15;
      ++v45;
      v20 = v47 + 1;
      if ( 4 * ((int)v47 + 1) >= (unsigned int)(3 * a2) )
        goto LABEL_14;
      if ( (int)a2 - (v20 + HIDWORD(v47)) <= (unsigned int)a2 >> 3 )
      {
        sub_AEE220((__int64)&v45, a2);
        if ( !(_DWORD)v48 )
        {
LABEL_84:
          LODWORD(v47) = v47 + 1;
          BUG();
        }
        v38 = 1;
        v39 = 0;
        v40 = (v48 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v13 = (_QWORD *)(v46 + 8LL * v40);
        v41 = *v13;
        v20 = v47 + 1;
        if ( v17 != (_BYTE *)*v13 )
        {
          while ( v41 != -4096 )
          {
            if ( !v39 && v41 == -8192 )
              v39 = v13;
            v40 = (v48 - 1) & (v38 + v40);
            v13 = (_QWORD *)(v46 + 8LL * v40);
            v41 = *v13;
            if ( v17 == (_BYTE *)*v13 )
              goto LABEL_16;
            ++v38;
          }
          if ( v39 )
            v13 = v39;
        }
      }
LABEL_16:
      LODWORD(v47) = v20;
      if ( *v13 != -4096 )
        --HIDWORD(v47);
      *v13 = v17;
      v21 = *(v17 - 16);
      if ( (v21 & 2) != 0 )
      {
        v22 = (_BYTE **)*((_QWORD *)v17 - 4);
        v23 = *((unsigned int *)v17 - 6);
      }
      else
      {
        v23 = (*((_WORD *)v17 - 8) >> 6) & 0xF;
        v22 = (_BYTE **)&v17[-8 * ((v21 >> 2) & 0xF) - 16];
      }
      v24 = &v22[v23];
      v10 = v46;
      a2 = (unsigned int)v48;
      v11 = (unsigned int)v50;
      while ( v24 != v22 )
      {
        v27 = *v22;
        if ( !*v22 || (unsigned __int8)(*v27 - 5) > 0x1Fu )
          goto LABEL_23;
        if ( (_DWORD)a2 )
        {
          v25 = (a2 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v26 = *(_BYTE **)(v10 + 8LL * v25);
          if ( v27 == v26 )
            goto LABEL_23;
          v42 = 1;
          while ( v26 != (_BYTE *)-4096LL )
          {
            v25 = (a2 - 1) & (v42 + v25);
            v26 = *(_BYTE **)(v10 + 8LL * v25);
            if ( v27 == v26 )
              goto LABEL_23;
            ++v42;
          }
        }
        v28 = *(_DWORD *)(a1 + 24);
        v29 = *(_QWORD *)(a1 + 8);
        if ( !v28 )
          goto LABEL_31;
        v30 = v28 - 1;
        v31 = v30 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v32 = *(_BYTE **)(v29 + 16LL * v31);
        if ( v27 != v32 )
        {
          v33 = 1;
          while ( v32 != (_BYTE *)-4096LL )
          {
            v31 = v30 & (v33 + v31);
            v32 = *(_BYTE **)(v29 + 16LL * v31);
            if ( v27 == v32 )
              goto LABEL_23;
            ++v33;
          }
LABEL_31:
          if ( *v17 != 18
            || ((v34 = *(v17 - 16), (v34 & 2) == 0)
              ? (v35 = (__int64)&v17[-8 * ((v34 >> 2) & 0xF) - 16])
              : (v35 = *((_QWORD *)v17 - 4)),
                (v36 = *(_BYTE **)(v35 + 56)) == 0 || v27 != v36) )
          {
            if ( *v27 != 17 )
            {
              if ( v11 + 1 > (unsigned __int64)HIDWORD(v50) )
              {
                sub_C8D5F0(&v49, v51, v11 + 1, 8);
                v11 = (unsigned int)v50;
              }
              v49[v11] = v27;
              a2 = (unsigned int)v48;
              v10 = v46;
              v11 = (unsigned int)(v50 + 1);
              LODWORD(v50) = v50 + 1;
            }
          }
        }
LABEL_23:
        ++v22;
      }
LABEL_10:
      if ( !(_DWORD)v11 )
      {
        v37 = 8 * a2;
        sub_C7D6A0(v10, v37, 8);
        if ( v49 != v51 )
          _libc_free(v49, v37);
        return;
      }
      v9 = v49;
    }
    ++v45;
LABEL_14:
    sub_AEE220((__int64)&v45, 2 * a2);
    if ( !(_DWORD)v48 )
      goto LABEL_84;
    v18 = (v48 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v13 = (_QWORD *)(v46 + 8LL * v18);
    v19 = *v13;
    v20 = v47 + 1;
    if ( v17 != (_BYTE *)*v13 )
    {
      v43 = 1;
      v44 = 0;
      while ( v19 != -4096 )
      {
        if ( v19 == -8192 && !v44 )
          v44 = v13;
        v18 = (v48 - 1) & (v43 + v18);
        v13 = (_QWORD *)(v46 + 8LL * v18);
        v19 = *v13;
        if ( v17 == (_BYTE *)*v13 )
          goto LABEL_16;
        ++v43;
      }
      if ( v44 )
        v13 = v44;
    }
    goto LABEL_16;
  }
}
