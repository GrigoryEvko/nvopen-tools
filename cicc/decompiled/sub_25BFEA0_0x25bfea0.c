// Function: sub_25BFEA0
// Address: 0x25bfea0
//
__int64 __fastcall sub_25BFEA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  int v5; // edx
  __int64 *v6; // rcx
  __int64 i; // rbx
  unsigned int v8; // esi
  __int64 *v9; // rax
  __int64 v10; // r10
  int v11; // eax
  __int64 v12; // rax
  unsigned int v13; // r8d
  __int64 v14; // rdi
  unsigned __int64 v15; // r12
  char v16; // r13
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rbx
  unsigned __int8 *v20; // rdx
  int v21; // eax
  unsigned __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v25; // r13
  __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  unsigned int v32; // edx
  __int64 v33; // rsi
  unsigned int v34; // ecx
  __int64 *v35; // rax
  __int64 v36; // r8
  int v37; // ecx
  int v38; // eax
  __int64 v39; // rdi
  unsigned int v40; // r8d
  unsigned int v41; // r10d
  __int64 *v42; // rax
  __int64 v43; // r11
  int v44; // eax
  int v45; // eax
  int v46; // r13d
  int v47; // eax
  int v48; // eax
  int v49; // r11d
  int v50; // r9d
  unsigned __int64 v52; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int64 v53; // [rsp+28h] [rbp-B8h]
  _DWORD v54[3]; // [rsp+30h] [rbp-B0h] BYREF
  char v55; // [rsp+3Ch] [rbp-A4h]
  _QWORD v56[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v57; // [rsp+50h] [rbp-90h] BYREF
  _BYTE *v58; // [rsp+58h] [rbp-88h]
  __int64 v59; // [rsp+60h] [rbp-80h]
  int v60; // [rsp+68h] [rbp-78h]
  char v61; // [rsp+6Ch] [rbp-74h]
  _BYTE v62[112]; // [rsp+70h] [rbp-70h] BYREF

  v4 = sub_BC0510(a4, qword_4F86C48, a3);
  v52 = (unsigned __int64)v54;
  v53 = 0x1000000000LL;
  sub_D2AD40(v4 + 8, qword_4F86C48);
  v5 = *(_DWORD *)(v4 + 448);
  if ( v5 )
  {
    v6 = *(__int64 **)(v4 + 440);
    for ( i = *v6; i; i = v6[v11] )
    {
      v12 = *(unsigned int *)(i + 16);
      if ( (_DWORD)v12 )
      {
LABEL_26:
        v25 = *(_QWORD *)(i + 8);
        v26 = v25 + 8 * v12;
        do
        {
          while ( 1 )
          {
            if ( *(_DWORD *)(*(_QWORD *)v25 + 16LL) == 1 )
            {
              v27 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)v25 + 8LL) + 8LL);
              if ( !sub_B2FC80(v27) && !(unsigned __int8)sub_B2D610(v27, 34) && (*(_BYTE *)(v27 + 32) & 0xF) == 7 )
                break;
            }
            v25 += 8;
            if ( v26 == v25 )
              goto LABEL_35;
          }
          v30 = (unsigned int)v53;
          v31 = (unsigned int)v53 + 1LL;
          if ( v31 > HIDWORD(v53) )
          {
            sub_C8D5F0((__int64)&v52, v54, v31, 8u, v28, v29);
            v30 = (unsigned int)v53;
          }
          v25 += 8;
          *(_QWORD *)(v52 + 8 * v30) = v27;
          LODWORD(v53) = v53 + 1;
        }
        while ( v26 != v25 );
LABEL_35:
        v32 = *(_DWORD *)(v4 + 608);
        v33 = *(_QWORD *)(v4 + 592);
        if ( v32 )
        {
          v34 = (v32 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v35 = (__int64 *)(v33 + 16LL * v34);
          v36 = *v35;
          if ( i == *v35 )
          {
LABEL_37:
            v37 = *(_DWORD *)(v4 + 448);
            v38 = *((_DWORD *)v35 + 2) + 1;
            if ( v38 == v37 )
              break;
            v39 = *(_QWORD *)(v4 + 440);
            i = *(_QWORD *)(v39 + 8LL * v38);
            if ( !i )
              break;
            v40 = v32 - 1;
LABEL_43:
            v12 = *(unsigned int *)(i + 16);
            if ( (_DWORD)v12 )
              goto LABEL_26;
            if ( v32 )
            {
              v41 = v40 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
              v42 = (__int64 *)(v33 + 16LL * v41);
              v43 = *v42;
              if ( *v42 != i )
              {
                v45 = 1;
                while ( v43 != -4096 )
                {
                  v46 = v45 + 1;
                  v41 = v40 & (v45 + v41);
                  v42 = (__int64 *)(v33 + 16LL * v41);
                  v43 = *v42;
                  if ( *v42 == i )
                    goto LABEL_41;
                  v45 = v46;
                }
                goto LABEL_45;
              }
            }
            else
            {
LABEL_45:
              v42 = (__int64 *)(v33 + 16LL * v32);
            }
LABEL_41:
            v44 = *((_DWORD *)v42 + 2) + 1;
            if ( v37 == v44 )
              break;
            i = *(_QWORD *)(v39 + 8LL * v44);
            if ( !i )
              break;
            goto LABEL_43;
          }
          v47 = 1;
          while ( v36 != -4096 )
          {
            v50 = v47 + 1;
            v34 = (v32 - 1) & (v47 + v34);
            v35 = (__int64 *)(v33 + 16LL * v34);
            v36 = *v35;
            if ( i == *v35 )
              goto LABEL_37;
            v47 = v50;
          }
        }
        v35 = (__int64 *)(v33 + 16LL * v32);
        goto LABEL_37;
      }
      v13 = *(_DWORD *)(v4 + 608);
      v14 = *(_QWORD *)(v4 + 592);
      if ( v13 )
      {
        v8 = (v13 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
        v9 = (__int64 *)(v14 + 16LL * v8);
        v10 = *v9;
        if ( i == *v9 )
        {
LABEL_5:
          v11 = *((_DWORD *)v9 + 2) + 1;
          if ( v11 == v5 )
            break;
          continue;
        }
        v48 = 1;
        while ( v10 != -4096 )
        {
          v49 = v48 + 1;
          v8 = (v13 - 1) & (v48 + v8);
          v9 = (__int64 *)(v14 + 16LL * v8);
          v10 = *v9;
          if ( *v9 == i )
            goto LABEL_5;
          v48 = v49;
        }
      }
      v11 = *(_DWORD *)(v14 + 16LL * v13 + 8) + 1;
      if ( v11 == v5 )
        break;
    }
  }
  v15 = v52;
  v16 = 0;
  v17 = v52 + 8LL * (unsigned int)v53;
  if ( v52 == v17 )
  {
    if ( (_DWORD *)v52 != v54 )
      _libc_free(v52);
LABEL_59:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 2;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    sub_AE6EC0(a1, (__int64)&qword_4F82400);
    return a1;
  }
  do
  {
    v18 = *(_QWORD *)(v17 - 8);
    v19 = *(_QWORD *)(v18 + 16);
    if ( v19 )
    {
      while ( 1 )
      {
        v20 = *(unsigned __int8 **)(v19 + 24);
        v21 = *v20;
        if ( (unsigned __int8)v21 <= 0x1Cu )
          break;
        v22 = (unsigned int)(v21 - 34);
        if ( (unsigned __int8)v22 > 0x33u )
          break;
        v23 = 0x8000000000041LL;
        if ( !_bittest64(&v23, v22)
          || (unsigned __int8 *)v19 != v20 - 32
          || !(unsigned __int8)sub_B2D610(*(_QWORD *)(*((_QWORD *)v20 + 5) + 72LL), 34) )
        {
          break;
        }
        v19 = *(_QWORD *)(v19 + 8);
        if ( !v19 )
          goto LABEL_18;
      }
    }
    else
    {
LABEL_18:
      v16 = 1;
      sub_B2CD30(v18, 34);
    }
    v17 -= 8;
  }
  while ( v15 != v17 );
  if ( (_DWORD *)v52 != v54 )
    _libc_free(v52);
  if ( !v16 )
    goto LABEL_59;
  v53 = (unsigned __int64)v56;
  v56[0] = qword_4F86C48;
  v54[0] = 2;
  v54[2] = 0;
  v55 = 1;
  v57 = 0;
  v58 = v62;
  v59 = 2;
  v60 = 0;
  v61 = 1;
  v54[1] = 1;
  v52 = 1;
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v56, (__int64)&v52);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v62, (__int64)&v57);
  if ( !v61 )
  {
    _libc_free((unsigned __int64)v58);
    if ( v55 )
      return a1;
    goto LABEL_60;
  }
  if ( !v55 )
LABEL_60:
    _libc_free(v53);
  return a1;
}
