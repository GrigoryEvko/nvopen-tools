// Function: sub_279F630
// Address: 0x279f630
//
__int64 __fastcall sub_279F630(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rcx
  int v5; // edx
  unsigned int v7; // r13d
  unsigned int v8; // eax
  __int64 v9; // rsi
  int k; // edi
  unsigned int v12; // eax
  unsigned int v13; // eax
  unsigned int v14; // ecx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  _QWORD *i; // rdx
  char *v19; // rax
  char *v20; // rbx
  _BYTE *v21; // r12
  char *v22; // r15
  _QWORD *v23; // rbx
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  unsigned __int8 **v29; // r15
  unsigned __int8 *v30; // r12
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  char *v39; // rax
  unsigned int v40; // eax
  unsigned int v41; // ebx
  char v42; // al
  __int64 v43; // rdi
  __int64 v44; // rax
  bool v45; // zf
  _QWORD *v46; // rax
  __int64 v47; // rdx
  _QWORD *j; // rdx
  _QWORD *v49; // rax
  __int64 v50; // rdx
  _QWORD *v51; // rdx
  unsigned __int64 v52; // [rsp+8h] [rbp-B8h]
  _QWORD *v53; // [rsp+10h] [rbp-B0h]
  _QWORD *v55; // [rsp+20h] [rbp-A0h]
  unsigned __int8 **v56; // [rsp+28h] [rbp-98h]
  __int64 v57; // [rsp+30h] [rbp-90h] BYREF
  char *v58; // [rsp+38h] [rbp-88h]
  __int64 v59; // [rsp+40h] [rbp-80h]
  int v60; // [rsp+48h] [rbp-78h]
  char v61; // [rsp+4Ch] [rbp-74h]
  char v62; // [rsp+50h] [rbp-70h] BYREF

  v3 = *(_DWORD *)(a1 + 72);
  v4 = *(_QWORD *)(a1 + 56);
  if ( !v3 )
  {
LABEL_6:
    v12 = *(_DWORD *)(a1 + 496);
    ++*(_QWORD *)(a1 + 488);
    v13 = v12 >> 1;
    if ( v13 )
    {
      if ( (*(_BYTE *)(a1 + 496) & 1) != 0 )
      {
LABEL_9:
        v15 = (_QWORD *)(a1 + 504);
        v16 = 8;
LABEL_15:
        for ( i = &v15[v16]; i != v15; v15 += 2 )
          *v15 = -4096;
        *(_QWORD *)(a1 + 496) &= 1uLL;
        goto LABEL_18;
      }
      v14 = 4 * v13;
    }
    else
    {
      if ( !*(_DWORD *)(a1 + 500) )
        goto LABEL_18;
      v14 = 0;
      if ( (*(_BYTE *)(a1 + 496) & 1) != 0 )
        goto LABEL_9;
    }
    v17 = *(unsigned int *)(a1 + 512);
    if ( (unsigned int)v17 <= v14 || (unsigned int)v17 <= 0x40 )
    {
      v15 = *(_QWORD **)(a1 + 504);
      v16 = 2 * v17;
      goto LABEL_15;
    }
    if ( v13 && (v40 = v13 - 1) != 0 )
    {
      _BitScanReverse(&v40, v40);
      v41 = 1 << (33 - (v40 ^ 0x1F));
      if ( v41 - 5 <= 0x3A )
      {
        v41 = 64;
        sub_C7D6A0(*(_QWORD *)(a1 + 504), 16 * v17, 8);
        v42 = *(_BYTE *)(a1 + 496);
        v43 = 1024;
LABEL_58:
        *(_BYTE *)(a1 + 496) = v42 & 0xFE;
        v44 = sub_C7D670(v43, 8);
        *(_DWORD *)(a1 + 512) = v41;
        *(_QWORD *)(a1 + 504) = v44;
        goto LABEL_59;
      }
      if ( (_DWORD)v17 == v41 )
      {
        v45 = (*(_QWORD *)(a1 + 496) & 1LL) == 0;
        *(_QWORD *)(a1 + 496) &= 1uLL;
        if ( v45 )
        {
          v49 = *(_QWORD **)(a1 + 504);
          v50 = 2 * v17;
        }
        else
        {
          v49 = (_QWORD *)(a1 + 504);
          v50 = 8;
        }
        v51 = &v49[v50];
        do
        {
          if ( v49 )
            *v49 = -4096;
          v49 += 2;
        }
        while ( v51 != v49 );
LABEL_18:
        *(_DWORD *)(a1 + 576) = 0;
        v57 = 0;
        v58 = &v62;
        v59 = 8;
        v60 = 0;
        v61 = 1;
        v7 = sub_F59E00(a2, (__int64)&v57);
        v19 = v58;
        if ( v61 )
          v20 = &v58[8 * HIDWORD(v59)];
        else
          v20 = &v58[8 * (unsigned int)v59];
        if ( v58 != v20 )
        {
          while ( 1 )
          {
            v21 = *(_BYTE **)v19;
            v22 = v19;
            if ( *(_QWORD *)v19 < 0xFFFFFFFFFFFFFFFELL )
              break;
            v19 += 8;
            if ( v20 == v19 )
              goto LABEL_23;
          }
          if ( v20 != v19 )
          {
            do
            {
              sub_278A7A0(a1 + 136, v21);
              sub_278C2C0((_QWORD *)a1, v21, v35, v36, v37, v38);
              v39 = v22 + 8;
              if ( v22 + 8 == v20 )
                break;
              v21 = *(_BYTE **)v39;
              for ( v22 += 8; *(_QWORD *)v39 >= 0xFFFFFFFFFFFFFFFELL; v22 = v39 )
              {
                v39 += 8;
                if ( v20 == v39 )
                  goto LABEL_23;
                v21 = *(_BYTE **)v39;
              }
            }
            while ( v22 != v20 );
          }
        }
LABEL_23:
        v23 = *(_QWORD **)(a2 + 56);
        v55 = (_QWORD *)(a2 + 48);
        if ( v23 == (_QWORD *)(a2 + 48) )
        {
LABEL_40:
          if ( !v61 )
            _libc_free((unsigned __int64)v58);
          return v7;
        }
        while ( 1 )
        {
          while ( 1 )
          {
            if ( *(_DWORD *)(a1 + 576) )
            {
              v24 = (__int64)(v23 - 3);
              if ( !v23 )
                v24 = 0;
              v7 |= sub_278BEC0(a1, v24);
            }
            v25 = (__int64)(v23 - 3);
            if ( !v23 )
              v25 = 0;
            v7 |= sub_279ECC0(a1, v25);
            v26 = *(unsigned int *)(a1 + 656);
            if ( (_DWORD)v26 )
              break;
            v23 = (_QWORD *)v23[1];
LABEL_26:
            if ( v55 == v23 )
              goto LABEL_40;
          }
          v27 = 8 * v26;
          v28 = *(_QWORD *)(a1 + 648);
          v53 = *(_QWORD **)(a2 + 56);
          if ( v53 == v23 )
          {
            v52 = (unsigned __int64)v23;
            v56 = (unsigned __int8 **)(v28 + v27);
          }
          else
          {
            v56 = (unsigned __int8 **)(v28 + v27);
            v52 = *v23 & 0xFFFFFFFFFFFFFFF8LL;
          }
          v29 = *(unsigned __int8 ***)(a1 + 648);
          do
          {
            v30 = *v29++;
            sub_11C4E30(v30, *(_QWORD *)(a1 + 40), 0);
            sub_F54ED0(v30);
            sub_278C2C0((_QWORD *)a1, v30, v31, v32, v33, v34);
          }
          while ( v56 != v29 );
          *(_DWORD *)(a1 + 656) = 0;
          if ( v53 == v23 )
          {
            v23 = *(_QWORD **)(a2 + 56);
            goto LABEL_26;
          }
          v23 = *(_QWORD **)(v52 + 8);
          if ( v55 == v23 )
            goto LABEL_40;
        }
      }
      sub_C7D6A0(*(_QWORD *)(a1 + 504), 16 * v17, 8);
      v42 = *(_BYTE *)(a1 + 496) | 1;
      *(_BYTE *)(a1 + 496) = v42;
      if ( v41 > 4 )
      {
        v43 = 16LL * v41;
        goto LABEL_58;
      }
    }
    else
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 504), 16 * v17, 8);
      *(_BYTE *)(a1 + 496) |= 1u;
    }
LABEL_59:
    v45 = (*(_QWORD *)(a1 + 496) & 1LL) == 0;
    *(_QWORD *)(a1 + 496) &= 1uLL;
    if ( v45 )
    {
      v46 = *(_QWORD **)(a1 + 504);
      v47 = 2LL * *(unsigned int *)(a1 + 512);
    }
    else
    {
      v46 = (_QWORD *)(a1 + 504);
      v47 = 8;
    }
    for ( j = &v46[v47]; j != v46; v46 += 2 )
    {
      if ( v46 )
        *v46 = -4096;
    }
    goto LABEL_18;
  }
  v5 = v3 - 1;
  v7 = 0;
  v8 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = *(_QWORD *)(v4 + 8LL * v8);
  if ( v9 != a2 )
  {
    for ( k = 1; ; ++k )
    {
      if ( v9 == -4096 )
        goto LABEL_6;
      v8 = v5 & (k + v8);
      v9 = *(_QWORD *)(v4 + 8LL * v8);
      if ( v9 == a2 )
        break;
    }
    return 0;
  }
  return v7;
}
