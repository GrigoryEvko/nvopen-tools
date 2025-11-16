// Function: sub_32AEA30
// Address: 0x32aea30
//
void __fastcall sub_32AEA30(__int64 a1, char *a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  int v6; // r13d
  __int64 j; // rcx
  _BYTE *v8; // rax
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned int v11; // r13d
  _BYTE *v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // r12
  __int64 i; // r14
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 v19; // r8
  void (__fastcall *v20)(_QWORD *, _QWORD, _QWORD); // rax
  int v21; // eax
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rbx
  int v24; // edi
  _BYTE *v25; // rsi
  __int64 v26; // rbx
  __int64 v27; // r14
  _QWORD *v28; // r13
  void (__fastcall *v29)(_QWORD *, __int64, _QWORD); // rax
  __int64 v30; // rsi
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 *v34; // rcx
  __int64 v35; // rsi
  int v36; // edi
  _BYTE *v37; // rsi
  __int64 v40; // [rsp+30h] [rbp-B0h]
  __int64 v42; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v43; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v44; // [rsp+40h] [rbp-A0h]
  int v45; // [rsp+48h] [rbp-98h]
  __int64 v46; // [rsp+48h] [rbp-98h]
  unsigned __int64 v47; // [rsp+48h] [rbp-98h]
  unsigned __int64 v48; // [rsp+48h] [rbp-98h]
  _BYTE *v49; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v50; // [rsp+58h] [rbp-88h]
  _BYTE v51[32]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v52; // [rsp+80h] [rbp-60h] BYREF
  __int64 v53; // [rsp+88h] [rbp-58h]
  _BYTE v54[80]; // [rsp+90h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a1 + 136);
  if ( v6 )
  {
    LODWORD(j) = *(_DWORD *)(a1 + 140);
    v49 = v51;
    v50 = 0x400000000LL;
    v52 = v54;
    v53 = 0x400000000LL;
    if ( (_DWORD)j )
    {
      v8 = v51;
      v9 = 1;
      j = 0;
      v10 = *(_QWORD *)(a1 + 8);
      while ( 1 )
      {
        *(_QWORD *)&v8[8 * j] = v10;
        j = (unsigned int)(v50 + 1);
        LODWORD(v50) = v50 + 1;
        if ( *(_DWORD *)(a1 + 140) == (_DWORD)v9 )
          break;
        v10 = *(_QWORD *)(a1 + 8 * v9 + 8);
        if ( j + 1 > (unsigned __int64)HIDWORD(v50) )
        {
          sub_C8D5F0((__int64)&v49, v51, j + 1, 8u, a5, a6);
          j = (unsigned int)v50;
        }
        v8 = v49;
        ++v9;
      }
      v45 = *(_DWORD *)(a1 + 136);
      v11 = v45 - 1;
      if ( v45 != 1 )
      {
        v12 = v49;
        v13 = (unsigned int)v53;
LABEL_10:
        v14 = (_QWORD *)(a3 + a1);
        v40 = (unsigned __int8)a2 & 1;
        if ( !(_DWORD)j )
          goto LABEL_27;
LABEL_11:
        v42 = 8LL * (unsigned int)(j - 1);
        for ( i = 0; ; i += 8 )
        {
          v16 = *(_QWORD *)&v12[i];
          v17 = 0;
          v18 = 8 * (v16 & 0x3F) + 8;
          while ( 1 )
          {
            v19 = *(_QWORD *)((v16 & 0xFFFFFFFFFFFFFFC0LL) + v17);
            if ( v13 + 1 > (unsigned __int64)HIDWORD(v53) )
            {
              v46 = *(_QWORD *)((v16 & 0xFFFFFFFFFFFFFFC0LL) + v17);
              sub_C8D5F0((__int64)&v52, v54, v13 + 1, 8u, v19, a6);
              v13 = (unsigned int)v53;
              v19 = v46;
            }
            v17 += 8;
            *(_QWORD *)&v52[8 * v13] = v19;
            v13 = (unsigned int)(v53 + 1);
            LODWORD(v53) = v53 + 1;
            if ( v18 == v17 )
              break;
            v16 = *(_QWORD *)&v49[i];
          }
          v20 = (void (__fastcall *)(_QWORD *, _QWORD, _QWORD))a2;
          if ( v40 )
            v20 = *(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))&a2[*v14 - 1];
          v20(v14, *(_QWORD *)&v49[i], v11);
          v12 = v49;
          if ( v42 == i )
            break;
          v13 = (unsigned int)v53;
        }
        for ( j = (unsigned int)v53; ; j = (unsigned int)v13 )
        {
          LODWORD(v50) = 0;
          v21 = HIDWORD(v50);
          if ( v12 != v51 )
          {
            v22 = (unsigned __int64)v52;
            if ( v52 != v54 )
              break;
          }
          a6 = (unsigned int)j;
          if ( HIDWORD(v50) < (unsigned int)j )
          {
            sub_C8D5F0((__int64)&v49, v51, (unsigned int)j, 8u, a5, (unsigned int)j);
            a5 = (unsigned int)v50;
            j = (unsigned int)v50;
            if ( HIDWORD(v53) < (unsigned int)v50 )
            {
              sub_C8D5F0((__int64)&v52, v54, (unsigned int)v50, 8u, (unsigned int)v50, v31);
              a5 = (unsigned int)v50;
              j = (unsigned int)v50;
            }
            a6 = (unsigned int)v53;
            v23 = (unsigned int)v53;
            if ( a5 <= (unsigned int)v53 )
              v23 = a5;
            if ( v23 )
            {
              v32 = 0;
              do
              {
                v33 = (__int64 *)&v52[v32];
                v34 = (__int64 *)&v49[v32];
                v32 += 8;
                v35 = *v34;
                *v34 = *v33;
                *v33 = v35;
              }
              while ( 8 * v23 != v32 );
              a5 = (unsigned int)v50;
              a6 = (unsigned int)v53;
              j = (unsigned int)v50;
            }
            if ( a5 > a6 )
            {
              v36 = a6;
              v37 = &v49[8 * v23];
              if ( v37 != &v49[8 * a5] )
              {
                v44 = a5;
                v48 = a6;
                memcpy(&v52[8 * a6], v37, 8 * a5 - 8 * v23);
                v36 = v53;
                a5 = v44;
                a6 = v48;
              }
              a5 -= a6;
              LODWORD(v50) = v23;
              j = (unsigned int)v23;
              LODWORD(v53) = a5 + v36;
LABEL_25:
              if ( !--v11 )
                goto LABEL_34;
              goto LABEL_26;
            }
          }
          else
          {
            a5 = 0;
            j = 0;
            v23 = 0;
          }
          if ( a5 >= a6 )
            goto LABEL_25;
          v24 = a5;
          v25 = &v52[8 * v23];
          if ( v25 != &v52[8 * a6] )
          {
            v43 = a6;
            v47 = a5;
            memcpy(&v49[8 * a5], v25, 8 * a6 - 8 * v23);
            v24 = v50;
            a6 = v43;
            a5 = v47;
          }
          a6 -= a5;
          LODWORD(v53) = v23;
          LODWORD(v50) = a6 + v24;
          j = (unsigned int)(a6 + v24);
          if ( !--v11 )
            goto LABEL_34;
LABEL_26:
          v12 = v49;
          v13 = (unsigned int)v53;
          if ( (_DWORD)j )
            goto LABEL_11;
LABEL_27:
          ;
        }
        v52 = v12;
        v49 = (_BYTE *)v22;
        v50 = __PAIR64__(HIDWORD(v53), j);
        LODWORD(v53) = 0;
        HIDWORD(v53) = v21;
        goto LABEL_25;
      }
LABEL_34:
      if ( (_DWORD)j )
      {
        v26 = 8 * j;
        v27 = 0;
        v28 = (_QWORD *)(a3 + a1);
        do
        {
          v29 = (void (__fastcall *)(_QWORD *, __int64, _QWORD))a2;
          if ( ((unsigned __int8)a2 & 1) != 0 )
            v29 = *(void (__fastcall **)(_QWORD *, __int64, _QWORD))&a2[*v28 - 1];
          v30 = *(_QWORD *)&v49[v27];
          v27 += 8;
          v29(v28, v30, 0);
        }
        while ( v26 != v27 );
      }
      if ( v52 != v54 )
        _libc_free((unsigned __int64)v52);
    }
    else
    {
      v11 = v6 - 1;
      if ( v11 )
      {
        v12 = v51;
        v13 = 0;
        goto LABEL_10;
      }
    }
    if ( v49 != v51 )
      _libc_free((unsigned __int64)v49);
  }
}
