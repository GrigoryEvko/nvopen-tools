// Function: sub_1B275E0
// Address: 0x1b275e0
//
void __fastcall sub_1B275E0(_QWORD **a1, __int64 a2, __int64 a3, __int64 ****a4, __int64 a5)
{
  int v6; // r8d
  __int64 v7; // r13
  __int64 *v8; // rax
  __int64 v9; // r10
  __int64 *v10; // rbx
  __int64 v11; // rdx
  __int64 *v12; // r9
  __int64 *v13; // r14
  __int64 *v14; // rdx
  char v15; // dl
  __int64 v16; // r15
  __int64 *v17; // rsi
  __int64 *v18; // rcx
  __int64 v19; // rax
  __int64 ****v20; // rbx
  __int64 *v21; // r13
  __int64 ****v22; // r14
  int v23; // r8d
  int v24; // r9d
  char v25; // dl
  __int64 v26; // r15
  __int64 *v27; // rax
  __int64 *v28; // rsi
  __int64 *v29; // rcx
  __int64 v30; // rax
  __int64 *v31; // rax
  __int128 v32; // rdi
  __int64 v33; // r14
  __int64 v34; // rcx
  __int64 v35; // rbx
  _QWORD *v36; // r13
  _QWORD v39[2]; // [rsp+20h] [rbp-1A0h] BYREF
  _QWORD *v40; // [rsp+30h] [rbp-190h] BYREF
  __int16 v41; // [rsp+40h] [rbp-180h]
  _BYTE *v42; // [rsp+50h] [rbp-170h] BYREF
  __int64 v43; // [rsp+58h] [rbp-168h]
  _BYTE v44[128]; // [rsp+60h] [rbp-160h] BYREF
  __int64 v45; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 *v46; // [rsp+E8h] [rbp-D8h]
  __int64 *v47; // [rsp+F0h] [rbp-D0h]
  __int64 v48; // [rsp+F8h] [rbp-C8h]
  int v49; // [rsp+100h] [rbp-C0h]
  _BYTE v50[184]; // [rsp+108h] [rbp-B8h] BYREF

  v39[0] = a2;
  v39[1] = a3;
  v42 = v44;
  v7 = sub_16321C0((__int64)a1, a2, a3, 0);
  v8 = (__int64 *)v50;
  v45 = 0;
  v46 = (__int64 *)v50;
  v47 = (__int64 *)v50;
  v48 = 16;
  v49 = 0;
  v43 = 0x1000000000LL;
  if ( v7 )
  {
    v9 = *(_QWORD *)(v7 - 24);
    if ( *(_BYTE *)(v9 + 16) != 6 )
      BUG();
    v10 = *(__int64 **)(v7 - 24);
    v11 = 3LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    v12 = (__int64 *)(v9 - v11 * 8);
    if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
    {
      v12 = *(__int64 **)(v9 - 8);
      v10 = &v12[v11];
    }
    if ( v12 != v10 )
    {
      v13 = v12;
      v14 = (__int64 *)v50;
      while ( 1 )
      {
        v16 = *v13;
        if ( v8 == v14 )
        {
          v17 = &v8[HIDWORD(v48)];
          if ( v17 != v8 )
          {
            v18 = 0;
            while ( v16 != *v8 )
            {
              if ( *v8 == -2 )
                v18 = v8;
              if ( v17 == ++v8 )
              {
                if ( !v18 )
                  goto LABEL_49;
                *v18 = v16;
                --v49;
                ++v45;
                goto LABEL_19;
              }
            }
LABEL_8:
            v13 += 3;
            if ( v10 == v13 )
              goto LABEL_21;
            goto LABEL_9;
          }
LABEL_49:
          if ( HIDWORD(v48) < (unsigned int)v48 )
            break;
        }
        sub_16CCBA0((__int64)&v45, *v13);
        if ( !v15 )
          goto LABEL_8;
LABEL_19:
        v19 = (unsigned int)v43;
        if ( (unsigned int)v43 >= HIDWORD(v43) )
          goto LABEL_51;
LABEL_20:
        v13 += 3;
        *(_QWORD *)&v42[8 * v19] = v16;
        LODWORD(v43) = v43 + 1;
        if ( v10 == v13 )
          goto LABEL_21;
LABEL_9:
        v14 = v47;
        v8 = v46;
      }
      ++HIDWORD(v48);
      *v17 = v16;
      v19 = (unsigned int)v43;
      ++v45;
      if ( (unsigned int)v43 < HIDWORD(v43) )
        goto LABEL_20;
LABEL_51:
      sub_16CD150((__int64)&v42, v44, 0, 8, v6, (int)v12);
      v19 = (unsigned int)v43;
      goto LABEL_20;
    }
LABEL_21:
    sub_15E55B0(v7);
  }
  v20 = a4;
  v21 = (__int64 *)sub_16471D0(*a1, 0);
  v22 = &a4[a5];
  if ( a4 != v22 )
  {
    do
    {
      while ( 1 )
      {
        v26 = sub_15A4AD0(*v20, (__int64)v21);
        v27 = v46;
        if ( v47 != v46 )
          break;
        v28 = &v46[HIDWORD(v48)];
        if ( v46 != v28 )
        {
          v29 = 0;
          while ( v26 != *v27 )
          {
            if ( *v27 == -2 )
              v29 = v27;
            if ( v28 == ++v27 )
            {
              if ( !v29 )
                goto LABEL_43;
              *v29 = v26;
              --v49;
              ++v45;
              goto LABEL_35;
            }
          }
          goto LABEL_25;
        }
LABEL_43:
        if ( HIDWORD(v48) >= (unsigned int)v48 )
          break;
        ++HIDWORD(v48);
        *v28 = v26;
        v30 = (unsigned int)v43;
        ++v45;
        if ( (unsigned int)v43 >= HIDWORD(v43) )
        {
LABEL_45:
          sub_16CD150((__int64)&v42, v44, 0, 8, v23, v24);
          v30 = (unsigned int)v43;
        }
LABEL_36:
        ++v20;
        *(_QWORD *)&v42[8 * v30] = v26;
        LODWORD(v43) = v43 + 1;
        if ( v22 == v20 )
          goto LABEL_37;
      }
      sub_16CCBA0((__int64)&v45, v26);
      if ( v25 )
      {
LABEL_35:
        v30 = (unsigned int)v43;
        if ( (unsigned int)v43 >= HIDWORD(v43) )
          goto LABEL_45;
        goto LABEL_36;
      }
LABEL_25:
      ++v20;
    }
    while ( v22 != v20 );
  }
LABEL_37:
  if ( (_DWORD)v43 )
  {
    v31 = sub_1645D80(v21, (unsigned int)v43);
    *((_QWORD *)&v32 + 1) = v42;
    *(_QWORD *)&v32 = v31;
    v33 = (__int64)v31;
    v35 = sub_159DFD0(v32, (unsigned int)v43, v34);
    v41 = 261;
    v40 = v39;
    v36 = sub_1648A60(88, 1u);
    if ( v36 )
      sub_15E51E0((__int64)v36, (__int64)a1, v33, 0, 6, v35, (__int64)&v40, 0, 0, 0, 0);
    sub_15E5D20((__int64)v36, "llvm.metadata", 0xDu);
  }
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
  if ( v47 != v46 )
    _libc_free((unsigned __int64)v47);
}
