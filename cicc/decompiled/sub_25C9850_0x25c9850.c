// Function: sub_25C9850
// Address: 0x25c9850
//
char __fastcall sub_25C9850(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // r14
  __int64 v4; // r12
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 **v7; // rbx
  __int64 **j; // r12
  __int64 *v9; // r13
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 i; // r13
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  int v18; // r12d
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 *v24; // rcx
  unsigned __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  int v30; // edx
  __int64 v31; // rsi
  int v32; // ecx
  unsigned int v33; // edx
  __int64 v34; // rdi
  __int64 v35; // r14
  __int64 v36; // r12
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  _QWORD *v43; // rdi
  _QWORD *v44; // rsi
  int v45; // r8d
  __int64 v49; // [rsp+10h] [rbp-120h]
  int v50; // [rsp+18h] [rbp-118h]
  char v51; // [rsp+1Eh] [rbp-112h]
  char v52; // [rsp+1Fh] [rbp-111h]
  __int64 *v53; // [rsp+28h] [rbp-108h]
  __int64 v54; // [rsp+30h] [rbp-100h]
  __int64 *v55; // [rsp+38h] [rbp-F8h]
  __m128i v56; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v57; // [rsp+50h] [rbp-E0h]
  __int64 v58; // [rsp+58h] [rbp-D8h]
  __int64 v59; // [rsp+60h] [rbp-D0h]
  __int64 v60; // [rsp+68h] [rbp-C8h]
  __int64 v61; // [rsp+70h] [rbp-C0h]
  __int64 v62; // [rsp+78h] [rbp-B8h]
  __int16 v63; // [rsp+80h] [rbp-B0h]
  __int64 v64; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v65; // [rsp+98h] [rbp-98h]
  __int64 v66; // [rsp+A0h] [rbp-90h]
  __int64 v67; // [rsp+A8h] [rbp-88h]
  _BYTE *v68; // [rsp+B0h] [rbp-80h]
  __int64 v69; // [rsp+B8h] [rbp-78h]
  _BYTE v70[112]; // [rsp+C0h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a1 + 32);
  v55 = (__int64 *)(v2 + 8LL * *(unsigned int *)(a1 + 40));
  if ( (__int64 *)v2 == v55 )
    return v2;
  v3 = *(__int64 **)(a1 + 32);
  v52 = 1;
  while ( 1 )
  {
LABEL_3:
    while ( 1 )
    {
      v4 = *v3;
      v64 = *(_QWORD *)(*v3 + 120);
      LOBYTE(v2) = sub_A74710(&v64, 0, 43);
      if ( !(_BYTE)v2 )
      {
        LOBYTE(v2) = sub_B2FC80(v4);
        if ( (_BYTE)v2 )
          return v2;
        LOBYTE(v2) = sub_B2FC00((_BYTE *)v4);
        v51 = v2;
        if ( (_BYTE)v2 )
          return v2;
        v2 = **(_QWORD **)(*(_QWORD *)(v4 + 24) + 16LL);
        if ( *(_BYTE *)(v2 + 8) == 14 )
          break;
      }
LABEL_8:
      if ( v55 == ++v3 )
        goto LABEL_9;
    }
    v64 = 0;
    v65 = 0;
    v68 = v70;
    v66 = 0;
    v67 = 0;
    v69 = 0x800000000LL;
    for ( i = *(_QWORD *)(v4 + 80); v4 + 72 != i; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      v14 = *(_QWORD *)(i + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v14 == i + 24 )
        goto LABEL_71;
      if ( !v14 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 > 0xA )
LABEL_71:
        BUG();
      if ( *(_BYTE *)(v14 - 24) == 30 )
      {
        v15 = 0;
        v16 = *(_DWORD *)(v14 - 20) & 0x7FFFFFF;
        if ( (*(_DWORD *)(v14 - 20) & 0x7FFFFFF) != 0 )
        {
          v16 = -32LL * (unsigned int)v16;
          v15 = *(_QWORD *)(v14 + v16 - 24);
        }
        v56.m128i_i64[0] = v15;
        sub_25C9420((__int64)&v64, v56.m128i_i64, v16, v15, v5, v6);
      }
    }
    v54 = sub_B2BEC0(v4);
    if ( (_DWORD)v69 )
      break;
    v25 = (unsigned __int64)v68;
    if ( v68 == v70 )
    {
      sub_C7D6A0(v65, 8LL * (unsigned int)v67, 8);
      goto LABEL_43;
    }
LABEL_41:
    _libc_free(v25);
LABEL_42:
    LOBYTE(v2) = sub_C7D6A0(v65, 8LL * (unsigned int)v67, 8);
    if ( v51 )
      goto LABEL_8;
LABEL_43:
    ++v3;
    sub_B2D390(v4, 43);
    LOBYTE(v2) = sub_25C0F40((__int64)&v64, a2, (__int64 *)v4, v26, v27, v28);
    if ( v55 == v3 )
      goto LABEL_9;
  }
  v53 = v3;
  v17 = 0;
  v49 = v4;
  v18 = 0;
  while ( 2 )
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)&v68[8 * v17];
      v57 = 0;
      v56 = (__m128i)(unsigned __int64)v54;
      v58 = 0;
      v59 = 0;
      v60 = 0;
      v61 = 0;
      v62 = 0;
      v63 = 257;
      if ( !(unsigned __int8)sub_9B6260(v19, &v56, 0) )
        break;
LABEL_39:
      v17 = (unsigned int)(v18 + 1);
      v18 = v17;
      if ( (_DWORD)v69 == (_DWORD)v17 )
        goto LABEL_40;
    }
    if ( *(_BYTE *)v19 <= 0x1Cu )
      goto LABEL_32;
    switch ( *(_BYTE *)v19 )
    {
      case '"':
      case 'U':
        v29 = *(_QWORD *)(v19 - 32);
        if ( !v29 || *(_BYTE *)v29 || *(_QWORD *)(v29 + 24) != *(_QWORD *)(v19 + 80) )
          goto LABEL_32;
        v56.m128i_i64[0] = *(_QWORD *)(v19 - 32);
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v43 = *(_QWORD **)(a1 + 32);
          v44 = &v43[*(unsigned int *)(a1 + 40)];
          if ( v44 == sub_25BD100(v43, (__int64)v44, v56.m128i_i64) )
            goto LABEL_32;
          goto LABEL_51;
        }
        v30 = *(_DWORD *)(a1 + 24);
        v31 = *(_QWORD *)(a1 + 8);
        if ( !v30 )
          goto LABEL_32;
        v32 = v30 - 1;
        v33 = (v30 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v34 = *(_QWORD *)(v31 + 8LL * v33);
        if ( v29 == v34 )
        {
LABEL_51:
          v17 = (unsigned int)(v18 + 1);
          v51 = 1;
          v18 = v17;
          if ( (_DWORD)v69 == (_DWORD)v17 )
          {
LABEL_40:
            v4 = v49;
            v3 = v53;
            v25 = (unsigned __int64)v68;
            if ( v68 == v70 )
              goto LABEL_42;
            goto LABEL_41;
          }
          continue;
        }
        v45 = 1;
        while ( v34 != -4096 )
        {
          v33 = v32 & (v45 + v33);
          v34 = *(_QWORD *)(v31 + 8LL * v33);
          if ( v29 == v34 )
            goto LABEL_51;
          ++v45;
        }
LABEL_32:
        if ( v68 != v70 )
          _libc_free((unsigned __int64)v68);
        v3 = v53 + 1;
        LOBYTE(v2) = sub_C7D6A0(v65, 8LL * (unsigned int)v67, 8);
        v52 = 0;
        if ( v55 != v53 + 1 )
          goto LABEL_3;
LABEL_9:
        if ( v52 )
        {
          v7 = *(__int64 ***)(a1 + 32);
          v2 = *(unsigned int *)(a1 + 40);
          for ( j = &v7[v2]; j != v7; ++v7 )
          {
            v9 = *v7;
            v64 = (*v7)[15];
            LOBYTE(v2) = sub_A74710(&v64, 0, 43);
            if ( !(_BYTE)v2 )
            {
              v2 = **(_QWORD **)(v9[3] + 16);
              if ( *(_BYTE *)(v2 + 8) == 14 )
              {
                sub_B2D390((__int64)v9, 43);
                LOBYTE(v2) = sub_25C0F40((__int64)&v64, a2, v9, v10, v11, v12);
              }
            }
          }
        }
        return v2;
      case '?':
        if ( (*(_BYTE *)(v19 + 1) & 2) == 0 )
          goto LABEL_32;
        goto LABEL_36;
      case 'N':
      case 'O':
LABEL_36:
        if ( (*(_BYTE *)(v19 + 7) & 0x40) != 0 )
          v24 = *(__int64 **)(v19 - 8);
        else
          v24 = (__int64 *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF));
        v56.m128i_i64[0] = *v24;
        sub_25C9420((__int64)&v64, v56.m128i_i64, v20, (__int64)v24, v22, v23);
        goto LABEL_39;
      case 'T':
        if ( (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) == 0 )
          goto LABEL_39;
        v50 = v18;
        v35 = 0;
        v36 = v19;
        v37 = 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF);
        do
        {
          v38 = *(_QWORD *)(*(_QWORD *)(v36 - 8) + v35);
          v35 += 32;
          v56.m128i_i64[0] = v38;
          sub_25C9420((__int64)&v64, v56.m128i_i64, v20, v21, v22, v23);
        }
        while ( v37 != v35 );
        v17 = (unsigned int)(v50 + 1);
        v18 = v17;
        if ( (_DWORD)v69 == (_DWORD)v17 )
          goto LABEL_40;
        continue;
      case 'V':
        v56.m128i_i64[0] = *(_QWORD *)(v19 - 64);
        sub_25C9420((__int64)&v64, v56.m128i_i64, v20, v21, v22, v23);
        v56.m128i_i64[0] = *(_QWORD *)(v19 - 32);
        sub_25C9420((__int64)&v64, v56.m128i_i64, v39, v40, v41, v42);
        v17 = (unsigned int)(v18 + 1);
        v18 = v17;
        if ( (_DWORD)v69 == (_DWORD)v17 )
          goto LABEL_40;
        continue;
      default:
        goto LABEL_32;
    }
  }
}
