// Function: sub_34D0EB0
// Address: 0x34d0eb0
//
unsigned __int64 __fastcall sub_34D0EB0(__int64 a1, _BYTE **a2, int a3, __int64 *a4, __int64 a5, __int64 a6)
{
  _BYTE **v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r15
  unsigned __int64 v10; // r14
  unsigned __int8 v11; // al
  __int64 v12; // r13
  _BYTE *v13; // rsi
  int v14; // edi
  __int64 v15; // r8
  __int64 v16; // rdx
  char *v17; // rax
  unsigned int v18; // eax
  char v19; // dl
  int v20; // edx
  int v22; // r8d
  unsigned int v23; // edi
  unsigned __int64 v24; // rdx
  bool v25; // zf
  __int64 v26; // rdx
  unsigned __int64 v27; // r14
  int v28; // r15d
  unsigned int i; // r12d
  unsigned __int64 v30; // rax
  __int64 *v31; // rsi
  unsigned int v32; // eax
  bool v33; // of
  unsigned __int64 v34; // rdi
  int v35; // r13d
  int v36; // eax
  __int64 *v37; // [rsp+0h] [rbp-B0h]
  __int64 v38; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v39; // [rsp+18h] [rbp-98h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  int v42; // [rsp+2Ch] [rbp-84h]
  unsigned __int64 v43; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v44; // [rsp+38h] [rbp-78h]
  __int64 v45; // [rsp+40h] [rbp-70h] BYREF
  char *v46; // [rsp+48h] [rbp-68h]
  __int64 v47; // [rsp+50h] [rbp-60h]
  int v48; // [rsp+58h] [rbp-58h]
  unsigned __int8 v49; // [rsp+5Ch] [rbp-54h]
  char v50; // [rsp+60h] [rbp-50h] BYREF

  v45 = 0;
  v46 = &v50;
  v47 = 4;
  v48 = 0;
  v49 = 1;
  if ( !a3 )
    return 0;
  v7 = a2;
  v8 = 1;
  v42 = 0;
  v9 = (__int64)&a2[(unsigned int)(a3 - 1) + 1];
  v10 = 0;
  do
  {
    while ( 1 )
    {
      v12 = *a4;
      v13 = *v7;
      v14 = *(unsigned __int8 *)(*a4 + 8);
      v15 = (unsigned int)(v14 - 17);
      v16 = *(unsigned __int8 *)(*a4 + 8);
      if ( (unsigned int)v15 > 1 )
      {
        if ( (_BYTE)v14 == 12 )
          goto LABEL_8;
LABEL_12:
        v11 = *(_BYTE *)(*a4 + 8);
        if ( v14 == 17 )
          v11 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
        goto LABEL_5;
      }
      v11 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
      if ( v11 == 12 )
        goto LABEL_8;
      if ( v14 != 18 )
        goto LABEL_12;
LABEL_5:
      if ( v11 > 3u && v11 != 5 && (v11 & 0xFD) != 4 )
        break;
LABEL_8:
      if ( *v13 > 0x15u )
        goto LABEL_18;
LABEL_9:
      ++v7;
      ++a4;
      if ( (_BYTE **)v9 == v7 )
        goto LABEL_27;
    }
    if ( (unsigned int)v15 <= 1 )
      v16 = *(unsigned __int8 *)(**(_QWORD **)(v12 + 16) + 8LL);
    if ( (_BYTE)v16 != 14 || *v13 <= 0x15u )
      goto LABEL_9;
LABEL_18:
    if ( !(_BYTE)v8 )
      goto LABEL_23;
    v17 = v46;
    v16 = (__int64)&v46[8 * HIDWORD(v47)];
    if ( v46 != (char *)v16 )
    {
      while ( v13 != *(_BYTE **)v17 )
      {
        v17 += 8;
        if ( (char *)v16 == v17 )
          goto LABEL_22;
      }
      goto LABEL_9;
    }
LABEL_22:
    if ( HIDWORD(v47) < (unsigned int)v47 )
    {
      ++HIDWORD(v47);
      *(_QWORD *)v16 = v13;
      v18 = v49;
      v20 = *(unsigned __int8 *)(v12 + 8);
      ++v45;
      v8 = v49;
      if ( v20 == 17 )
        goto LABEL_31;
    }
    else
    {
LABEL_23:
      sub_C8CC70((__int64)&v45, (__int64)v13, v16, v8, v15, a6);
      v18 = v49;
      v8 = v49;
      if ( !v19 )
        goto LABEL_9;
      v20 = *(unsigned __int8 *)(v12 + 8);
      if ( v20 == 17 )
      {
LABEL_31:
        v22 = *(_DWORD *)(v12 + 32);
        v44 = v22;
        v23 = v22;
        if ( (unsigned int)v22 > 0x40 )
        {
          sub_C43690((__int64)&v43, -1, 1);
          v23 = v44;
          if ( *(_BYTE *)(v12 + 8) != 18 )
          {
            v22 = *(_DWORD *)(v12 + 32);
LABEL_35:
            v26 = 0;
            if ( v22 > 0 )
            {
              v39 = v10;
              a6 = 1;
              v27 = 0;
              v38 = v9;
              v28 = v22;
              v37 = a4;
              for ( i = 0; i != v28; ++i )
              {
                v30 = v43;
                if ( v23 > 0x40 )
                  v30 = *(_QWORD *)(v43 + 8LL * (i >> 6));
                if ( (v30 & (1LL << i)) != 0 )
                {
                  v31 = (__int64 *)v12;
                  if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
                    v31 = **(__int64 ***)(v12 + 16);
                  v32 = sub_34D06B0(a1, v31);
                  a6 = 1;
                  if ( __OFADD__(v32, v27) )
                  {
                    v27 = 0x8000000000000000LL;
                    v23 = v44;
                    if ( v32 )
                      v27 = 0x7FFFFFFFFFFFFFFFLL;
                  }
                  else
                  {
                    v23 = v44;
                    v27 += v32;
                  }
                }
              }
              v26 = v27;
              v9 = v38;
              a4 = v37;
              v10 = v39;
            }
            if ( v23 > 0x40 && (v34 = v43) != 0 )
            {
              v35 = 0;
LABEL_57:
              v40 = v26;
              j_j___libc_free_0_0(v34);
              v8 = v49;
              v36 = 1;
              if ( v35 != 1 )
                v36 = v42;
              v26 = v40;
              v42 = v36;
            }
            else
            {
              v8 = v49;
            }
            goto LABEL_48;
          }
          if ( v44 > 0x40 )
          {
            v34 = v43;
            if ( v43 )
            {
              v35 = 1;
              v26 = 0;
              goto LABEL_57;
            }
            v42 = 1;
            v8 = v49;
            v26 = 0;
LABEL_48:
            v33 = __OFADD__(v26, v10);
            v10 += v26;
            if ( v33 )
            {
              v10 = 0x8000000000000000LL;
              if ( v26 > 0 )
                v10 = 0x7FFFFFFFFFFFFFFFLL;
            }
            goto LABEL_9;
          }
          v18 = v49;
        }
        else
        {
          v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
          if ( !v22 )
            v24 = 0;
          v25 = *(_BYTE *)(v12 + 8) == 18;
          v43 = v24;
          if ( !v25 )
            goto LABEL_35;
        }
        v8 = v18;
        goto LABEL_26;
      }
    }
    if ( v20 != 18 )
      goto LABEL_9;
LABEL_26:
    ++v7;
    ++a4;
    v42 = 1;
  }
  while ( (_BYTE **)v9 != v7 );
LABEL_27:
  if ( !(_BYTE)v8 )
    _libc_free((unsigned __int64)v46);
  return v10;
}
