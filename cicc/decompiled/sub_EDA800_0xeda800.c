// Function: sub_EDA800
// Address: 0xeda800
//
void __fastcall sub_EDA800(__int64 a1, char *a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  int v6; // r13d
  __int64 j; // rcx
  _BYTE *v8; // rsi
  _BYTE *v9; // rax
  __int64 v10; // r12
  __int64 v11; // r13
  unsigned int v12; // r13d
  _BYTE *v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // r12
  __int64 i; // r14
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // r8
  void (__fastcall *v21)(_QWORD *, _BYTE *, _QWORD); // rax
  int v22; // eax
  unsigned __int64 v23; // rbx
  int v24; // edi
  __int64 v25; // rbx
  __int64 v26; // r14
  _QWORD *v27; // r13
  void (__fastcall *v28)(_QWORD *, _BYTE *, _QWORD); // rax
  __int64 v29; // r9
  __int64 v30; // rax
  _QWORD *v31; // rdx
  _QWORD *v32; // rcx
  int v33; // edi
  __int64 v37; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v38; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v39; // [rsp+40h] [rbp-A0h]
  int v40; // [rsp+48h] [rbp-98h]
  __int64 v41; // [rsp+48h] [rbp-98h]
  unsigned __int64 v42; // [rsp+48h] [rbp-98h]
  unsigned __int64 v43; // [rsp+48h] [rbp-98h]
  _BYTE *v44; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v45; // [rsp+58h] [rbp-88h]
  _BYTE v46[32]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v47; // [rsp+80h] [rbp-60h] BYREF
  __int64 v48; // [rsp+88h] [rbp-58h]
  _BYTE v49[80]; // [rsp+90h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a1 + 96);
  if ( v6 )
  {
    LODWORD(j) = *(_DWORD *)(a1 + 100);
    v8 = v49;
    v44 = v46;
    v45 = 0x400000000LL;
    v47 = v49;
    v48 = 0x400000000LL;
    if ( (_DWORD)j )
    {
      v9 = v46;
      v10 = 1;
      j = 0;
      v11 = *(_QWORD *)(a1 + 8);
      while ( 1 )
      {
        *(_QWORD *)&v9[8 * j] = v11;
        j = (unsigned int)(v45 + 1);
        LODWORD(v45) = v45 + 1;
        if ( *(_DWORD *)(a1 + 100) == (_DWORD)v10 )
          break;
        v11 = *(_QWORD *)(a1 + 8 * v10 + 8);
        if ( j + 1 > (unsigned __int64)HIDWORD(v45) )
        {
          v8 = v46;
          sub_C8D5F0((__int64)&v44, v46, j + 1, 8u, a5, a6);
          j = (unsigned int)v45;
        }
        v9 = v44;
        ++v10;
      }
      v40 = *(_DWORD *)(a1 + 96);
      v12 = v40 - 1;
      if ( v40 != 1 )
      {
        v13 = v44;
        v14 = (unsigned int)v48;
LABEL_10:
        v15 = (_QWORD *)(a3 + a1);
        if ( !(_DWORD)j )
          goto LABEL_27;
LABEL_11:
        v37 = 8LL * (unsigned int)(j - 1);
        for ( i = 0; ; i += 8 )
        {
          v17 = *(_QWORD *)&v13[i];
          v18 = 0;
          v19 = 8 * (v17 & 0x3F) + 8;
          while ( 1 )
          {
            v20 = *(_QWORD *)((v17 & 0xFFFFFFFFFFFFFFC0LL) + v18);
            if ( v14 + 1 > (unsigned __int64)HIDWORD(v48) )
            {
              v41 = *(_QWORD *)((v17 & 0xFFFFFFFFFFFFFFC0LL) + v18);
              sub_C8D5F0((__int64)&v47, v49, v14 + 1, 8u, v20, a6);
              v14 = (unsigned int)v48;
              v20 = v41;
            }
            v18 += 8;
            *(_QWORD *)&v47[8 * v14] = v20;
            v14 = (unsigned int)(v48 + 1);
            LODWORD(v48) = v48 + 1;
            if ( v19 == v18 )
              break;
            v17 = *(_QWORD *)&v44[i];
          }
          v21 = (void (__fastcall *)(_QWORD *, _BYTE *, _QWORD))a2;
          if ( ((unsigned __int8)a2 & 1) != 0 )
            v21 = *(void (__fastcall **)(_QWORD *, _BYTE *, _QWORD))&a2[*v15 - 1];
          v8 = *(_BYTE **)&v44[i];
          v21(v15, v8, v12);
          v13 = v44;
          if ( v37 == i )
            break;
          v14 = (unsigned int)v48;
        }
        for ( j = (unsigned int)v48; ; j = (unsigned int)v14 )
        {
          LODWORD(v45) = 0;
          v22 = HIDWORD(v45);
          if ( v13 != v46 )
          {
            v8 = v47;
            if ( v47 != v49 )
              break;
          }
          a6 = (unsigned int)j;
          if ( HIDWORD(v45) < (unsigned int)j )
          {
            v8 = v46;
            sub_C8D5F0((__int64)&v44, v46, (unsigned int)j, 8u, a5, (unsigned int)j);
            a5 = (unsigned int)v45;
            j = (unsigned int)v45;
            if ( HIDWORD(v48) < (unsigned int)v45 )
            {
              v8 = v49;
              sub_C8D5F0((__int64)&v47, v49, (unsigned int)v45, 8u, (unsigned int)v45, v29);
              a5 = (unsigned int)v45;
              j = (unsigned int)v45;
            }
            a6 = (unsigned int)v48;
            v23 = (unsigned int)v48;
            if ( a5 <= (unsigned int)v48 )
              v23 = a5;
            if ( v23 )
            {
              v30 = 0;
              do
              {
                v31 = &v47[v30];
                v32 = &v44[v30];
                v30 += 8;
                v8 = (_BYTE *)*v32;
                *v32 = *v31;
                *v31 = v8;
              }
              while ( 8 * v23 != v30 );
              a5 = (unsigned int)v45;
              a6 = (unsigned int)v48;
              j = (unsigned int)v45;
            }
            if ( a5 > a6 )
            {
              v33 = a6;
              v8 = &v44[8 * v23];
              if ( v8 != &v44[8 * a5] )
              {
                v39 = a5;
                v43 = a6;
                memcpy(&v47[8 * a6], v8, 8 * a5 - 8 * v23);
                v33 = v48;
                a5 = v39;
                a6 = v43;
              }
              a5 -= a6;
              LODWORD(v45) = v23;
              j = (unsigned int)v23;
              LODWORD(v48) = a5 + v33;
LABEL_25:
              if ( !--v12 )
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
          v8 = &v47[8 * v23];
          if ( v8 != &v47[8 * a6] )
          {
            v38 = a6;
            v42 = a5;
            memcpy(&v44[8 * a5], v8, 8 * a6 - 8 * v23);
            v24 = v45;
            a6 = v38;
            a5 = v42;
          }
          a6 -= a5;
          LODWORD(v48) = v23;
          LODWORD(v45) = a6 + v24;
          j = (unsigned int)(a6 + v24);
          if ( !--v12 )
            goto LABEL_34;
LABEL_26:
          v13 = v44;
          v14 = (unsigned int)v48;
          if ( (_DWORD)j )
            goto LABEL_11;
LABEL_27:
          ;
        }
        v47 = v13;
        v44 = v8;
        v45 = __PAIR64__(HIDWORD(v48), j);
        LODWORD(v48) = 0;
        HIDWORD(v48) = v22;
        goto LABEL_25;
      }
LABEL_34:
      if ( (_DWORD)j )
      {
        v25 = 8 * j;
        v26 = 0;
        v27 = (_QWORD *)(a3 + a1);
        do
        {
          v28 = (void (__fastcall *)(_QWORD *, _BYTE *, _QWORD))a2;
          if ( ((unsigned __int8)a2 & 1) != 0 )
            v28 = *(void (__fastcall **)(_QWORD *, _BYTE *, _QWORD))&a2[*v27 - 1];
          v8 = *(_BYTE **)&v44[v26];
          v26 += 8;
          v28(v27, v8, 0);
        }
        while ( v25 != v26 );
      }
      if ( v47 != v49 )
        _libc_free(v47, v8);
    }
    else
    {
      v12 = v6 - 1;
      if ( v12 )
      {
        v13 = v46;
        v14 = 0;
        goto LABEL_10;
      }
    }
    if ( v44 != v46 )
      _libc_free(v44, v8);
  }
}
