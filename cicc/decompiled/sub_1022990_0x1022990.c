// Function: sub_1022990
// Address: 0x1022990
//
__int64 __fastcall sub_1022990(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  __int64 v8; // r9
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r8
  char v18; // al
  __int64 v19; // rsi
  char v20; // di
  char *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  char v24; // dl
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r12
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rbx
  unsigned __int8 *v33; // r14
  unsigned __int8 **v34; // rax
  char v35; // al
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 *v38; // rax
  __int64 v39; // [rsp+18h] [rbp-128h]
  __int64 v40; // [rsp+18h] [rbp-128h]
  __int64 v41; // [rsp+18h] [rbp-128h]
  __int64 v42; // [rsp+18h] [rbp-128h]
  _BYTE *v43; // [rsp+18h] [rbp-128h]
  __int64 v44; // [rsp+20h] [rbp-120h] BYREF
  char *v45; // [rsp+28h] [rbp-118h]
  __int64 v46; // [rsp+30h] [rbp-110h]
  int v47; // [rsp+38h] [rbp-108h]
  char v48; // [rsp+3Ch] [rbp-104h]
  char v49; // [rsp+40h] [rbp-100h] BYREF
  _QWORD *v50; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v51; // [rsp+68h] [rbp-D8h]
  _QWORD v52[8]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v53; // [rsp+B0h] [rbp-90h] BYREF
  unsigned __int8 **v54; // [rsp+B8h] [rbp-88h]
  __int64 v55; // [rsp+C0h] [rbp-80h]
  int v56; // [rsp+C8h] [rbp-78h]
  char v57; // [rsp+CCh] [rbp-74h]
  char v58; // [rsp+D0h] [rbp-70h] BYREF

  if ( *(_QWORD *)(a1 + 40) != **(_QWORD **)(a2 + 32) )
    return 0;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 )
    return 0;
  v39 = sub_D4B130(a2);
  v9 = sub_D47930(a2);
  LOBYTE(v3) = v9 == 0 || v39 == 0;
  if ( (_BYTE)v3 )
    return 0;
  v10 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 0 )
    return 0;
  v11 = *(_QWORD *)(a1 - 8);
  v12 = 0;
  v13 = v11 + 32LL * *(unsigned int *)(a1 + 72);
  while ( v39 != *(_QWORD *)(v13 + 8 * v12) )
  {
    if ( (_DWORD)v10 == (_DWORD)++v12 )
      return 0;
  }
  if ( (int)v12 < 0 )
    return 0;
  v14 = 0;
  while ( v9 != *(_QWORD *)(v13 + 8 * v14) )
  {
    if ( (_DWORD)v10 == (_DWORD)++v14 )
      return 0;
  }
  if ( (int)v14 < 0 )
    return 0;
  v15 = 0;
  do
  {
    if ( v9 == *(_QWORD *)(v13 + 8 * v15) )
    {
      v16 = 32 * v15;
      goto LABEL_19;
    }
    ++v15;
  }
  while ( (_DWORD)v10 != (_DWORD)v15 );
  v16 = 0x1FFFFFFFE0LL;
LABEL_19:
  v17 = *(_QWORD *)(v11 + v16);
  if ( *(_BYTE *)v17 <= 0x1Cu )
    return v3;
  v44 = 0;
  v45 = &v49;
  v46 = 4;
  v47 = 0;
  v48 = 1;
  v18 = *(_BYTE *)v17;
  while ( 1 )
  {
    v19 = *(_QWORD *)(v17 + 40);
    if ( v18 != 84 )
    {
      if ( *(_BYTE *)(a2 + 84) )
      {
        v25 = *(_QWORD **)(a2 + 64);
        v26 = &v25[*(unsigned int *)(a2 + 76)];
        if ( v25 == v26 )
        {
LABEL_60:
          v20 = v48;
          goto LABEL_35;
        }
        while ( v19 != *v25 )
        {
          if ( v26 == ++v25 )
            goto LABEL_60;
        }
      }
      else
      {
        v43 = (_BYTE *)v17;
        v38 = sub_C8CA60(a2 + 56, v19);
        v17 = (__int64)v43;
        if ( !v38 || *v43 == 84 )
        {
          v20 = v48;
          v3 = 0;
          goto LABEL_35;
        }
      }
      v27 = (__int64)v52;
      v28 = *(_QWORD *)(a1 + 40);
      v57 = 1;
      v52[0] = a1;
      v53 = 0;
      v55 = 8;
      v56 = 0;
      v50 = v52;
      v54 = (unsigned __int8 **)&v58;
      v51 = 0x800000001LL;
      v29 = 1;
      while ( 1 )
      {
        v30 = v29--;
        v31 = *(_QWORD *)(v27 + 8 * v30 - 8);
        LODWORD(v51) = v29;
        v32 = *(_QWORD *)(v31 + 16);
        if ( v32 )
          break;
LABEL_54:
        if ( !v29 )
        {
          v3 = 1;
LABEL_56:
          if ( v50 != v52 )
            _libc_free(v50, v19);
          if ( !v57 )
            _libc_free(v54, v19);
          goto LABEL_60;
        }
      }
      while ( 1 )
      {
        v33 = *(unsigned __int8 **)(v32 + 24);
        if ( v33 == (unsigned __int8 *)v17 )
          goto LABEL_56;
        if ( v57 )
        {
          v34 = v54;
          v27 = HIDWORD(v55);
          v31 = (__int64)&v54[HIDWORD(v55)];
          if ( v54 != (unsigned __int8 **)v31 )
          {
            while ( v33 != *v34 )
            {
              if ( (unsigned __int8 **)v31 == ++v34 )
                goto LABEL_71;
            }
            goto LABEL_52;
          }
LABEL_71:
          if ( HIDWORD(v55) < (unsigned int)v55 )
          {
            ++HIDWORD(v55);
            *(_QWORD *)v31 = v33;
            ++v53;
LABEL_62:
            v19 = v17;
            v42 = v17;
            v35 = sub_B19DB0(a3, v17, (__int64)v33);
            v17 = v42;
            if ( !v35 )
            {
              if ( v28 != *((_QWORD *)v33 + 5) )
                goto LABEL_56;
              if ( (unsigned __int8)sub_B46970(v33) )
                goto LABEL_56;
              if ( (unsigned __int8)sub_B46420((__int64)v33) )
                goto LABEL_56;
              v31 = *v33;
              if ( (unsigned int)(v31 - 30) <= 0xA )
                goto LABEL_56;
              v17 = v42;
              if ( (_BYTE)v31 != 84 )
              {
                v36 = (unsigned int)v51;
                v27 = HIDWORD(v51);
                v37 = (unsigned int)v51 + 1LL;
                if ( v37 > HIDWORD(v51) )
                {
                  v19 = (__int64)v52;
                  sub_C8D5F0((__int64)&v50, v52, v37, 8u, v42, v8);
                  v36 = (unsigned int)v51;
                  v17 = v42;
                }
                v31 = (__int64)v50;
                v50[v36] = v33;
                LODWORD(v51) = v51 + 1;
              }
            }
            goto LABEL_52;
          }
        }
        v19 = *(_QWORD *)(v32 + 24);
        v41 = v17;
        sub_C8CC70((__int64)&v53, v19, v31, v27, v17, v8);
        v17 = v41;
        if ( (_BYTE)v31 )
          goto LABEL_62;
LABEL_52:
        v32 = *(_QWORD *)(v32 + 8);
        if ( !v32 )
        {
          v29 = v51;
          v27 = (__int64)v50;
          goto LABEL_54;
        }
      }
    }
    v20 = v48;
    if ( *(_QWORD *)(a1 + 40) != v19 )
      goto LABEL_35;
    if ( !v48 )
      break;
    v21 = v45;
    v10 = (__int64)&v45[8 * HIDWORD(v46)];
    if ( v45 != (char *)v10 )
    {
      while ( *(_QWORD *)v21 != v17 )
      {
        v21 += 8;
        if ( (char *)v10 == v21 )
          goto LABEL_27;
      }
      return v3;
    }
LABEL_27:
    if ( HIDWORD(v46) >= (unsigned int)v46 )
      break;
    v19 = (unsigned int)++HIDWORD(v46);
    *(_QWORD *)v10 = v17;
    v20 = v48;
    ++v44;
LABEL_29:
    v8 = *(_QWORD *)(v17 - 8);
    v22 = 0x1FFFFFFFE0LL;
    v10 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
    if ( (*(_DWORD *)(v17 + 4) & 0x7FFFFFF) != 0 )
    {
      v23 = 0;
      v19 = v8 + 32LL * *(unsigned int *)(v17 + 72);
      do
      {
        if ( v9 == *(_QWORD *)(v19 + 8 * v23) )
        {
          v22 = 32 * v23;
          goto LABEL_34;
        }
        ++v23;
      }
      while ( (_DWORD)v10 != (_DWORD)v23 );
      v22 = 0x1FFFFFFFE0LL;
    }
LABEL_34:
    v17 = *(_QWORD *)(v8 + v22);
    v18 = *(_BYTE *)v17;
    if ( *(_BYTE *)v17 <= 0x1Cu )
      goto LABEL_35;
  }
  v19 = v17;
  v40 = v17;
  sub_C8CC70((__int64)&v44, v17, v10, 0x1FFFFFFFE0LL, v17, v8);
  v20 = v48;
  v17 = v40;
  if ( v24 )
    goto LABEL_29;
  v3 = 0;
LABEL_35:
  if ( !v20 )
    _libc_free(v45, v19);
  return v3;
}
