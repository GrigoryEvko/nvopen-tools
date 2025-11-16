// Function: sub_2DF5350
// Address: 0x2df5350
//
void __fastcall sub_2DF5350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  int v6; // ebx
  __int64 i; // rcx
  __int64 v9; // r13
  _BYTE *v10; // rax
  __int64 v11; // rbx
  unsigned int v12; // ebx
  _BYTE *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // r11
  int v20; // eax
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // r13
  int v23; // edi
  _BYTE *v24; // rsi
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rsi
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 *v31; // rcx
  __int64 v32; // rsi
  int v33; // edi
  _BYTE *v34; // rsi
  __int64 v35; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v36; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v37; // [rsp+20h] [rbp-A0h]
  __int64 v38; // [rsp+28h] [rbp-98h]
  unsigned __int64 v39; // [rsp+28h] [rbp-98h]
  unsigned __int64 v40; // [rsp+28h] [rbp-98h]
  _BYTE *v41; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v42; // [rsp+38h] [rbp-88h]
  _BYTE v43[32]; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v44; // [rsp+60h] [rbp-60h] BYREF
  __int64 v45; // [rsp+68h] [rbp-58h]
  _BYTE v46[80]; // [rsp+70h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a1 + 160);
  if ( v6 )
  {
    LODWORD(i) = *(_DWORD *)(a1 + 164);
    v41 = v43;
    v42 = 0x400000000LL;
    v44 = v46;
    v45 = 0x400000000LL;
    if ( (_DWORD)i )
    {
      v9 = *(_QWORD *)(a1 + 8);
      v10 = v43;
      v11 = 1;
      i = 0;
      while ( 1 )
      {
        *(_QWORD *)&v10[8 * i] = v9;
        i = (unsigned int)(v42 + 1);
        LODWORD(v42) = v42 + 1;
        if ( *(_DWORD *)(a1 + 164) == (_DWORD)v11 )
          break;
        v9 = *(_QWORD *)(a1 + 8 * v11 + 8);
        if ( i + 1 > (unsigned __int64)HIDWORD(v42) )
        {
          sub_C8D5F0((__int64)&v41, v43, i + 1, 8u, a5, a6);
          i = (unsigned int)v42;
        }
        v10 = v41;
        ++v11;
      }
      v12 = *(_DWORD *)(a1 + 160) - 1;
      if ( *(_DWORD *)(a1 + 160) != 1 )
      {
        v13 = v41;
        v14 = (unsigned int)v45;
LABEL_10:
        if ( !(_DWORD)i )
          goto LABEL_25;
LABEL_11:
        v15 = 0;
        v35 = 8LL * (unsigned int)(i - 1);
        while ( 1 )
        {
          v16 = *(_QWORD *)&v13[v15];
          v17 = 0;
          v18 = 8 * (v16 & 0x3F) + 8;
          while ( 1 )
          {
            v19 = *(_QWORD *)((v16 & 0xFFFFFFFFFFFFFFC0LL) + v17);
            if ( v14 + 1 > (unsigned __int64)HIDWORD(v45) )
            {
              v38 = *(_QWORD *)((v16 & 0xFFFFFFFFFFFFFFC0LL) + v17);
              sub_C8D5F0((__int64)&v44, v46, v14 + 1, 8u, a5, a6);
              v14 = (unsigned int)v45;
              v19 = v38;
            }
            v17 += 8;
            *(_QWORD *)&v44[8 * v14] = v19;
            v14 = (unsigned int)(v45 + 1);
            LODWORD(v45) = v45 + 1;
            if ( v18 == v17 )
              break;
            v16 = *(_QWORD *)&v41[v15];
          }
          sub_2DF57F0(a1, *(_QWORD *)&v41[v15], v12);
          v13 = v41;
          if ( v35 == v15 )
            break;
          v14 = (unsigned int)v45;
          v15 += 8;
        }
        for ( i = (unsigned int)v45; ; i = (unsigned int)v14 )
        {
          LODWORD(v42) = 0;
          v20 = HIDWORD(v42);
          if ( v13 != v43 )
          {
            v21 = (unsigned __int64)v44;
            if ( v44 != v46 )
              break;
          }
          a6 = (unsigned int)i;
          if ( HIDWORD(v42) < (unsigned int)i )
          {
            sub_C8D5F0((__int64)&v41, v43, (unsigned int)i, 8u, a5, (unsigned int)i);
            a5 = (unsigned int)v42;
            i = (unsigned int)v42;
            if ( HIDWORD(v45) < (unsigned int)v42 )
            {
              sub_C8D5F0((__int64)&v44, v46, (unsigned int)v42, 8u, (unsigned int)v42, v28);
              a5 = (unsigned int)v42;
              i = (unsigned int)v42;
            }
            a6 = (unsigned int)v45;
            v22 = (unsigned int)v45;
            if ( a5 <= (unsigned int)v45 )
              v22 = a5;
            if ( v22 )
            {
              v29 = 0;
              do
              {
                v30 = (__int64 *)&v44[v29];
                v31 = (__int64 *)&v41[v29];
                v29 += 8;
                v32 = *v31;
                *v31 = *v30;
                *v30 = v32;
              }
              while ( 8 * v22 != v29 );
              a5 = (unsigned int)v42;
              a6 = (unsigned int)v45;
              i = (unsigned int)v42;
            }
            if ( a5 > a6 )
            {
              v33 = a6;
              v34 = &v41[8 * v22];
              if ( v34 != &v41[8 * a5] )
              {
                v37 = a5;
                v40 = a6;
                memcpy(&v44[8 * a6], v34, 8 * a5 - 8 * v22);
                v33 = v45;
                a5 = v37;
                a6 = v40;
              }
              a5 -= a6;
              LODWORD(v42) = v22;
              i = (unsigned int)v22;
              LODWORD(v45) = a5 + v33;
LABEL_23:
              if ( !--v12 )
                goto LABEL_32;
              goto LABEL_24;
            }
          }
          else
          {
            a5 = 0;
            i = 0;
            v22 = 0;
          }
          if ( a5 >= a6 )
            goto LABEL_23;
          v23 = a5;
          v24 = &v44[8 * v22];
          if ( v24 != &v44[8 * a6] )
          {
            v36 = a6;
            v39 = a5;
            memcpy(&v41[8 * a5], v24, 8 * a6 - 8 * v22);
            v23 = v42;
            a6 = v36;
            a5 = v39;
          }
          a6 -= a5;
          LODWORD(v45) = v22;
          LODWORD(v42) = a6 + v23;
          i = (unsigned int)(a6 + v23);
          if ( !--v12 )
            goto LABEL_32;
LABEL_24:
          v13 = v41;
          v14 = (unsigned int)v45;
          if ( (_DWORD)i )
            goto LABEL_11;
LABEL_25:
          ;
        }
        v44 = v13;
        v41 = (_BYTE *)v21;
        v42 = __PAIR64__(HIDWORD(v45), i);
        LODWORD(v45) = 0;
        HIDWORD(v45) = v20;
        goto LABEL_23;
      }
LABEL_32:
      if ( (_DWORD)i )
      {
        v25 = 8 * i;
        v26 = 0;
        do
        {
          v27 = *(_QWORD *)&v41[v26];
          v26 += 8;
          sub_2DF57F0(a1, v27, 0);
        }
        while ( v25 != v26 );
      }
      if ( v44 != v46 )
        _libc_free((unsigned __int64)v44);
    }
    else
    {
      v12 = v6 - 1;
      if ( v12 )
      {
        v13 = v43;
        v14 = 0;
        goto LABEL_10;
      }
    }
    if ( v41 != v43 )
      _libc_free((unsigned __int64)v41);
  }
}
