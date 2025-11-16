// Function: sub_29EBCE0
// Address: 0x29ebce0
//
__int64 __fastcall sub_29EBCE0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  __int64 j; // r12
  __int64 v4; // r15
  size_t v5; // rdx
  char *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 *v12; // r8
  size_t v13; // rdx
  char *v14; // rsi
  size_t v15; // rax
  __int64 v16; // rax
  const char *v18; // rax
  size_t v19; // rdx
  size_t v20; // rbx
  const char *v21; // r12
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 i; // r14
  __int64 v27; // r8
  size_t v28; // rdx
  char *v29; // rsi
  __int64 v30; // r12
  __int64 v31; // rax
  size_t v32; // rdx
  __int64 *v33; // rcx
  int v34; // edx
  unsigned __int8 v35; // dl
  __int64 *v36; // rax
  unsigned int v37; // esi
  __int64 v38; // r9
  int v39; // r8d
  __int64 *v40; // rdi
  int v41; // r8d
  unsigned int v42; // esi
  __int64 v43; // r9
  __int64 v44; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+10h] [rbp-70h]
  unsigned int v46; // [rsp+10h] [rbp-70h]
  int v48; // [rsp+20h] [rbp-60h]
  __int64 *v49; // [rsp+20h] [rbp-60h]
  __int64 v50; // [rsp+20h] [rbp-60h]
  __int64 v51; // [rsp+20h] [rbp-60h]
  __int64 v52; // [rsp+30h] [rbp-50h] BYREF
  __int64 v53; // [rsp+38h] [rbp-48h]
  __int64 v54; // [rsp+40h] [rbp-40h]
  unsigned int v55; // [rsp+48h] [rbp-38h]

  v1 = a1 + 72;
  if ( !(unsigned __int8)sub_CEF7D0(2, a1) )
  {
    v18 = sub_BD5D20(a1);
    v20 = v19;
    v21 = v18;
    v22 = (__int64 *)sub_B2BE50(a1);
    v23 = sub_B9B140(v22, v21, v20);
    v24 = sub_29E0320(a1, v23);
    v25 = *(_QWORD *)(a1 + 80);
    v45 = v24;
    if ( v1 == v25 )
    {
      i = 0;
    }
    else
    {
      while ( 1 )
      {
        if ( !v25 )
LABEL_97:
          BUG();
        i = *(_QWORD *)(v25 + 32);
        if ( i != v25 + 24 )
          break;
        v25 = *(_QWORD *)(v25 + 8);
        if ( v1 == v25 )
          return sub_CEF870(2, a1);
      }
    }
    if ( v1 == v25 )
      return sub_CEF870(2, a1);
    while ( 1 )
    {
      v27 = i - 24;
      if ( !i )
        v27 = 0;
      v28 = 0;
      v29 = off_4C5D0D0[0];
      v30 = v27;
      if ( off_4C5D0D0[0] )
      {
        v29 = off_4C5D0D0[0];
        v28 = strlen(off_4C5D0D0[0]);
      }
      if ( *(_QWORD *)(v30 + 48) || (*(_BYTE *)(v30 + 7) & 0x20) != 0 )
      {
        if ( sub_B91F50(v30, v29, v28) )
          goto LABEL_43;
        v29 = off_4C5D0D0[0];
      }
      v32 = 0;
      if ( v29 )
        v32 = strlen(v29);
      sub_B9A090(v30, v29, v32, v45);
LABEL_43:
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v25 + 32) )
      {
        v31 = v25 - 24;
        if ( !v25 )
          v31 = 0;
        if ( i != v31 + 48 )
          break;
        v25 = *(_QWORD *)(v25 + 8);
        if ( v1 == v25 )
          return sub_CEF870(2, a1);
        if ( !v25 )
          goto LABEL_97;
      }
      if ( v1 == v25 )
        return sub_CEF870(2, a1);
    }
  }
  v2 = *(_QWORD *)(a1 + 80);
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  if ( v2 != v1 )
  {
    do
    {
      if ( !v2 )
        goto LABEL_97;
      j = *(_QWORD *)(v2 + 32);
      if ( j != v2 + 24 )
        break;
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v2 != v1 );
    if ( v1 != v2 )
    {
      while ( 1 )
      {
        v4 = j - 24;
        if ( !j )
          v4 = 0;
        v5 = 0;
        v6 = off_4C5D0D0[0];
        if ( off_4C5D0D0[0] )
        {
          v6 = off_4C5D0D0[0];
          v5 = strlen(off_4C5D0D0[0]);
        }
        if ( *(_QWORD *)(v4 + 48) || (*(_BYTE *)(v4 + 7) & 0x20) != 0 )
        {
          v7 = sub_B91F50(v4, v6, v5);
          if ( v7 )
            break;
        }
LABEL_19:
        for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v2 + 32) )
        {
          v16 = v2 - 24;
          if ( !v2 )
            v16 = 0;
          if ( j != v16 + 48 )
            break;
          v2 = *(_QWORD *)(v2 + 8);
          if ( v1 == v2 )
            return sub_C7D6A0(v53, 16LL * v55, 8);
          if ( !v2 )
            goto LABEL_97;
        }
        if ( v1 == v2 )
          return sub_C7D6A0(v53, 16LL * v55, 8);
      }
      if ( v55 )
      {
        v8 = (v55 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v9 = (__int64 *)(v53 + 16 * v8);
        v10 = *v9;
        if ( v7 == *v9 )
        {
LABEL_15:
          v11 = v9[1];
          v12 = v9 + 1;
          if ( v11 )
          {
LABEL_16:
            v13 = 0;
            v14 = off_4C5D0D0[0];
            if ( off_4C5D0D0[0] )
            {
              v44 = v11;
              v15 = strlen(off_4C5D0D0[0]);
              v11 = v44;
              v13 = v15;
            }
            sub_B9A090(v4, v14, v13, v11);
            goto LABEL_19;
          }
LABEL_66:
          v35 = *(_BYTE *)(v7 - 16);
          if ( (v35 & 2) != 0 )
            v36 = *(__int64 **)(v7 - 32);
          else
            v36 = (__int64 *)(v7 - 8LL * ((v35 >> 2) & 0xF) - 16);
          v49 = v12;
          v11 = sub_29E0320(a1, *v36);
          *v49 = v11;
          goto LABEL_16;
        }
        v48 = 1;
        v33 = 0;
        while ( v10 != -4096 )
        {
          if ( !v33 && v10 == -8192 )
            v33 = v9;
          LODWORD(v8) = (v55 - 1) & (v48 + v8);
          v9 = (__int64 *)(v53 + 16LL * (unsigned int)v8);
          v10 = *v9;
          if ( v7 == *v9 )
            goto LABEL_15;
          ++v48;
        }
        if ( !v33 )
          v33 = v9;
        ++v52;
        v34 = v54 + 1;
        if ( 4 * ((int)v54 + 1) < 3 * v55 )
        {
          if ( v55 - HIDWORD(v54) - v34 > v55 >> 3 )
            goto LABEL_63;
          v46 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
          v51 = v7;
          sub_AEABB0((__int64)&v52, v55);
          if ( !v55 )
          {
LABEL_98:
            LODWORD(v54) = v54 + 1;
            BUG();
          }
          v40 = 0;
          v41 = 1;
          v42 = (v55 - 1) & v46;
          v33 = (__int64 *)(v53 + 16LL * v42);
          v43 = *v33;
          v34 = v54 + 1;
          v7 = v51;
          if ( *v33 == v51 )
            goto LABEL_63;
          while ( v43 != -4096 )
          {
            if ( !v40 && v43 == -8192 )
              v40 = v33;
            v42 = (v55 - 1) & (v41 + v42);
            v33 = (__int64 *)(v53 + 16LL * v42);
            v43 = *v33;
            if ( v51 == *v33 )
              goto LABEL_63;
            ++v41;
          }
          goto LABEL_84;
        }
      }
      else
      {
        ++v52;
      }
      v50 = v7;
      sub_AEABB0((__int64)&v52, 2 * v55);
      if ( !v55 )
        goto LABEL_98;
      v7 = v50;
      v34 = v54 + 1;
      v37 = (v55 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
      v33 = (__int64 *)(v53 + 16LL * v37);
      v38 = *v33;
      if ( v50 == *v33 )
        goto LABEL_63;
      v39 = 1;
      v40 = 0;
      while ( v38 != -4096 )
      {
        if ( !v40 && v38 == -8192 )
          v40 = v33;
        v37 = (v55 - 1) & (v39 + v37);
        v33 = (__int64 *)(v53 + 16LL * v37);
        v38 = *v33;
        if ( v50 == *v33 )
          goto LABEL_63;
        ++v39;
      }
LABEL_84:
      if ( v40 )
        v33 = v40;
LABEL_63:
      LODWORD(v54) = v34;
      if ( *v33 != -4096 )
        --HIDWORD(v54);
      *v33 = v7;
      v12 = v33 + 1;
      v33[1] = 0;
      goto LABEL_66;
    }
  }
  return sub_C7D6A0(v53, 16LL * v55, 8);
}
