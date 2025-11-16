// Function: sub_1DA9BF0
// Address: 0x1da9bf0
//
void __fastcall sub_1DA9BF0(__int64 a1, char *a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  int v6; // r13d
  __int64 k; // rcx
  unsigned int v8; // r12d
  _BYTE *v9; // rax
  _QWORD *v10; // r13
  unsigned int v11; // r13d
  _BYTE *v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // r12
  __int64 i; // r14
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // r15
  __int64 j; // rbx
  _QWORD *v20; // rbx
  void (__fastcall *v21)(_QWORD *, _QWORD, _QWORD); // rax
  int v22; // eax
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rbx
  int v25; // edi
  _BYTE *v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // r14
  _QWORD *v29; // r13
  void (__fastcall *v30)(_QWORD *, __int64, _QWORD); // rax
  __int64 v31; // rsi
  int v32; // r9d
  __int64 v33; // rax
  __int64 *v34; // rdx
  __int64 *v35; // rcx
  __int64 v36; // rsi
  int v37; // edi
  _BYTE *v38; // rsi
  __int64 v41; // [rsp+30h] [rbp-B0h]
  __int64 v43; // [rsp+40h] [rbp-A0h]
  int v44; // [rsp+40h] [rbp-A0h]
  int v45; // [rsp+40h] [rbp-A0h]
  int v46; // [rsp+48h] [rbp-98h]
  _QWORD *v47; // [rsp+48h] [rbp-98h]
  int v48; // [rsp+48h] [rbp-98h]
  int v49; // [rsp+48h] [rbp-98h]
  _BYTE *v50; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v51; // [rsp+58h] [rbp-88h]
  _BYTE v52[32]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v53; // [rsp+80h] [rbp-60h] BYREF
  __int64 v54; // [rsp+88h] [rbp-58h]
  _BYTE v55[80]; // [rsp+90h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a1 + 80);
  if ( v6 )
  {
    LODWORD(k) = *(_DWORD *)(a1 + 84);
    v50 = v52;
    v51 = 0x400000000LL;
    v53 = v55;
    v54 = 0x400000000LL;
    if ( (_DWORD)k )
    {
      k = 0;
      v8 = 0;
      v9 = v52;
      v10 = (_QWORD *)(a1 + 8);
      while ( 1 )
      {
        ++v8;
        *(_QWORD *)&v9[8 * k] = *v10;
        k = (unsigned int)(v51 + 1);
        LODWORD(v51) = v51 + 1;
        if ( *(_DWORD *)(a1 + 84) == v8 )
          break;
        v10 = (_QWORD *)(a1 + 8 + 8LL * v8);
        if ( HIDWORD(v51) <= (unsigned int)k )
        {
          sub_16CD150((__int64)&v50, v52, 0, 8, a5, a6);
          k = (unsigned int)v51;
        }
        v9 = v50;
      }
      v46 = *(_DWORD *)(a1 + 80);
      v11 = v46 - 1;
      if ( v46 != 1 )
      {
        v12 = v50;
        v13 = (unsigned int)v54;
LABEL_10:
        v14 = (_QWORD *)(a3 + a1);
        v41 = (unsigned __int8)a2 & 1;
        if ( !(_DWORD)k )
          goto LABEL_27;
LABEL_11:
        v43 = 8LL * (unsigned int)(k - 1);
        for ( i = 0; ; i += 8 )
        {
          v16 = *(_QWORD *)&v12[i];
          v47 = v14;
          v17 = 8 * (v16 & 0x3F) + 8;
          v18 = 0;
          for ( j = v16; ; j = *(_QWORD *)&v50[i] )
          {
            v20 = (_QWORD *)(v18 + (j & 0xFFFFFFFFFFFFFFC0LL));
            if ( HIDWORD(v54) <= (unsigned int)v13 )
            {
              sub_16CD150((__int64)&v53, v55, 0, 8, a5, a6);
              v13 = (unsigned int)v54;
            }
            v18 += 8;
            *(_QWORD *)&v53[8 * v13] = *v20;
            v13 = (unsigned int)(v54 + 1);
            LODWORD(v54) = v54 + 1;
            if ( v17 == v18 )
              break;
          }
          v14 = v47;
          v21 = (void (__fastcall *)(_QWORD *, _QWORD, _QWORD))a2;
          if ( v41 )
            v21 = *(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))&a2[*v47 - 1];
          v21(v47, *(_QWORD *)&v50[i], v11);
          v12 = v50;
          if ( v43 == i )
            break;
          v13 = (unsigned int)v54;
        }
        for ( k = (unsigned int)v54; ; k = (unsigned int)v13 )
        {
          LODWORD(v51) = 0;
          v22 = HIDWORD(v51);
          if ( v12 != v52 )
          {
            v23 = (unsigned __int64)v53;
            if ( v53 != v55 )
              break;
          }
          a6 = (unsigned int)k;
          if ( HIDWORD(v51) < (unsigned int)k )
          {
            sub_16CD150((__int64)&v50, v52, (unsigned int)k, 8, a5, k);
            a5 = (unsigned int)v51;
            k = (unsigned int)v51;
            if ( HIDWORD(v54) < (unsigned int)v51 )
            {
              sub_16CD150((__int64)&v53, v55, (unsigned int)v51, 8, v51, v32);
              a5 = (unsigned int)v51;
              k = (unsigned int)v51;
            }
            a6 = (unsigned int)v54;
            v24 = (unsigned int)v54;
            if ( a5 <= (unsigned int)v54 )
              v24 = a5;
            if ( v24 )
            {
              v33 = 0;
              do
              {
                v34 = (__int64 *)&v53[v33];
                v35 = (__int64 *)&v50[v33];
                v33 += 8;
                v36 = *v35;
                *v35 = *v34;
                *v34 = v36;
              }
              while ( 8 * v24 != v33 );
              a5 = (unsigned int)v51;
              a6 = (unsigned int)v54;
              k = (unsigned int)v51;
            }
            if ( a5 > a6 )
            {
              v37 = a6;
              v38 = &v50[8 * v24];
              if ( v38 != &v50[8 * a5] )
              {
                v45 = a5;
                v49 = a6;
                memcpy(&v53[8 * a6], v38, 8 * a5 - 8 * v24);
                v37 = v54;
                LODWORD(a5) = v45;
                LODWORD(a6) = v49;
              }
              LODWORD(a5) = a5 - a6;
              LODWORD(v51) = v24;
              k = (unsigned int)v24;
              LODWORD(v54) = a5 + v37;
LABEL_25:
              if ( !--v11 )
                goto LABEL_34;
              goto LABEL_26;
            }
          }
          else
          {
            a5 = 0;
            k = 0;
            v24 = 0;
          }
          if ( a6 <= a5 )
            goto LABEL_25;
          v25 = a5;
          v26 = &v53[8 * v24];
          if ( v26 != &v53[8 * a6] )
          {
            v44 = a6;
            v48 = a5;
            memcpy(&v50[8 * a5], v26, 8 * a6 - 8 * v24);
            v25 = v51;
            LODWORD(a6) = v44;
            LODWORD(a5) = v48;
          }
          LODWORD(a6) = a6 - a5;
          LODWORD(v54) = v24;
          LODWORD(v51) = a6 + v25;
          k = (unsigned int)(a6 + v25);
          if ( !--v11 )
            goto LABEL_34;
LABEL_26:
          v12 = v50;
          v13 = (unsigned int)v54;
          if ( (_DWORD)k )
            goto LABEL_11;
LABEL_27:
          ;
        }
        v53 = v12;
        v50 = (_BYTE *)v23;
        v51 = __PAIR64__(HIDWORD(v54), k);
        LODWORD(v54) = 0;
        HIDWORD(v54) = v22;
        goto LABEL_25;
      }
LABEL_34:
      if ( (_DWORD)k )
      {
        v27 = 8 * k;
        v28 = 0;
        v29 = (_QWORD *)(a3 + a1);
        do
        {
          v30 = (void (__fastcall *)(_QWORD *, __int64, _QWORD))a2;
          if ( ((unsigned __int8)a2 & 1) != 0 )
            v30 = *(void (__fastcall **)(_QWORD *, __int64, _QWORD))&a2[*v29 - 1];
          v31 = *(_QWORD *)&v50[v28];
          v28 += 8;
          v30(v29, v31, 0);
        }
        while ( v27 != v28 );
      }
      if ( v53 != v55 )
        _libc_free((unsigned __int64)v53);
    }
    else
    {
      v11 = v6 - 1;
      if ( v11 )
      {
        v12 = v52;
        v13 = 0;
        goto LABEL_10;
      }
    }
    if ( v50 != v52 )
      _libc_free((unsigned __int64)v50);
  }
}
