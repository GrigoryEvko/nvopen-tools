// Function: sub_390EBE0
// Address: 0x390ebe0
//
__int64 __fastcall sub_390EBE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // r13
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 *v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // rax
  unsigned int v18; // esi
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // r10
  _QWORD *v22; // r9
  unsigned int v23; // r15d
  int v24; // ecx
  _QWORD *v25; // rax
  __int64 v26; // rdi
  __int64 *v27; // rax
  int v28; // r10d
  int v29; // r10d
  __int64 v30; // r11
  unsigned int v31; // esi
  __int64 v32; // rcx
  __int64 v33; // r9
  int v34; // edi
  int v35; // r9d
  __int64 v36; // r10
  _QWORD *v37; // r11
  unsigned int v38; // esi
  int v39; // edi
  _QWORD *v40; // rdi
  _BYTE *v41; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v42; // [rsp+8h] [rbp-D8h]
  _BYTE v43[64]; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v44; // [rsp+50h] [rbp-90h]
  char *v45; // [rsp+58h] [rbp-88h] BYREF
  __int64 v46; // [rsp+60h] [rbp-80h]
  _BYTE v47[120]; // [rsp+68h] [rbp-78h] BYREF

  v8 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v8 )
  {
    LODWORD(a5) = v8 - 1;
    v9 = *(_QWORD *)(a1 + 104);
    a4 = ((_DWORD)v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    a3 = v9 + 88 * a4;
    v10 = *(_QWORD *)a3;
    if ( a2 == *(_QWORD *)a3 )
    {
LABEL_3:
      a4 = 5 * v8;
      if ( a3 != v9 + 88 * v8 )
        return a3 + 8;
    }
    else
    {
      a3 = 1;
      while ( v10 != -8 )
      {
        a6 = a3 + 1;
        a4 = (unsigned int)a5 & ((_DWORD)a3 + (_DWORD)a4);
        a3 = v9 + 88LL * (unsigned int)a4;
        v10 = *(_QWORD *)a3;
        if ( a2 == *(_QWORD *)a3 )
          goto LABEL_3;
        a3 = a6;
      }
    }
  }
  v13 = a2;
  v14 = 0;
  v41 = v43;
  v42 = 0x800000000LL;
  while ( v13 )
  {
    if ( *(_BYTE *)(v13 + 16) == 9 )
    {
      if ( v13 != a2 && *(_BYTE *)(v13 + 56) )
        break;
      v15 = *(__int64 **)(a1 + 32);
      if ( v15 == *(__int64 **)(a1 + 24) )
        a3 = *(unsigned int *)(a1 + 44);
      else
        a3 = *(unsigned int *)(a1 + 40);
      v16 = &v15[a3];
      if ( v15 != v16 )
      {
        while ( 1 )
        {
          a3 = *v15;
          a4 = (__int64)v15;
          if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v16 == ++v15 )
            goto LABEL_18;
        }
        if ( v16 != v15 )
        {
          a5 = *(_QWORD *)(v13 + 48);
          if ( (*(_QWORD *)(a3 + 8) & a5) != 0 )
          {
LABEL_36:
            if ( (unsigned int)v14 >= HIDWORD(v42) )
            {
              sub_16CD150((__int64)&v41, v43, 0, 8, a5, a6);
              v14 = (unsigned int)v42;
            }
            *(_QWORD *)&v41[8 * v14] = v13;
            v14 = (unsigned int)(v42 + 1);
            LODWORD(v42) = v42 + 1;
          }
          else
          {
            while ( 1 )
            {
              v27 = (__int64 *)(a4 + 8);
              if ( (__int64 *)(a4 + 8) == v16 )
                break;
              a3 = *v27;
              for ( a4 += 8; (unsigned __int64)*v27 >= 0xFFFFFFFFFFFFFFFELL; a4 = (__int64)v27 )
              {
                if ( v16 == ++v27 )
                  goto LABEL_18;
                a3 = *v27;
              }
              if ( v16 == (__int64 *)a4 )
                break;
              if ( (*(_QWORD *)(a3 + 8) & a5) != 0 )
                goto LABEL_36;
            }
          }
        }
      }
    }
LABEL_18:
    v17 = *(_QWORD *)(v13 + 24);
    v13 = *(_QWORD *)(v13 + 8);
    if ( v17 + 96 == v13 )
      break;
  }
  v44 = a2;
  v45 = v47;
  v46 = 0x800000000LL;
  if ( (_DWORD)v14 )
    sub_390DB40((__int64)&v45, (__int64)&v41, a3, a4, a5, a6);
  v18 = *(_DWORD *)(a1 + 120);
  LODWORD(v19) = a1 + 96;
  if ( !v18 )
  {
    ++*(_QWORD *)(a1 + 96);
    goto LABEL_41;
  }
  v20 = v44;
  v21 = *(_QWORD *)(a1 + 104);
  v22 = 0;
  v23 = (v18 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
  v24 = 1;
  v25 = (_QWORD *)(v21 + 88LL * v23);
  v26 = *v25;
  if ( *v25 != v44 )
  {
    while ( v26 != -8 )
    {
      if ( !v22 && v26 == -16 )
        v22 = v25;
      v23 = (v18 - 1) & (v23 + v24);
      v25 = (_QWORD *)(v21 + 88LL * v23);
      v26 = *v25;
      if ( v44 == *v25 )
        goto LABEL_23;
      ++v24;
    }
    v34 = *(_DWORD *)(a1 + 112);
    if ( v22 )
      v25 = v22;
    ++*(_QWORD *)(a1 + 96);
    v32 = (unsigned int)(v34 + 1);
    if ( 4 * (int)v32 < 3 * v18 )
    {
      LODWORD(v33) = v18 >> 3;
      if ( v18 - *(_DWORD *)(a1 + 116) - (unsigned int)v32 > v18 >> 3 )
      {
LABEL_43:
        *(_DWORD *)(a1 + 112) = v32;
        if ( *v25 != -8 )
          --*(_DWORD *)(a1 + 116);
        *v25 = v20;
        v11 = (__int64)(v25 + 1);
        v25[1] = v25 + 3;
        v25[2] = 0x800000000LL;
        if ( (_DWORD)v46 )
          sub_390DA00(v11, &v45, (__int64)(v25 + 3), v32, v19, v33);
        goto LABEL_24;
      }
      sub_390E9A0(a1 + 96, v18);
      v35 = *(_DWORD *)(a1 + 120);
      if ( v35 )
      {
        LODWORD(v33) = v35 - 1;
        v36 = *(_QWORD *)(a1 + 104);
        v37 = 0;
        v20 = v44;
        v38 = v33 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v32 = (unsigned int)(*(_DWORD *)(a1 + 112) + 1);
        v39 = 1;
        v25 = (_QWORD *)(v36 + 88LL * v38);
        v19 = *v25;
        if ( *v25 != v44 )
        {
          while ( v19 != -8 )
          {
            if ( !v37 && v19 == -16 )
              v37 = v25;
            v38 = v33 & (v39 + v38);
            v25 = (_QWORD *)(v36 + 88LL * v38);
            v19 = *v25;
            if ( v44 == *v25 )
              goto LABEL_43;
            ++v39;
          }
          if ( v37 )
            v25 = v37;
        }
        goto LABEL_43;
      }
LABEL_80:
      ++*(_DWORD *)(a1 + 112);
      BUG();
    }
LABEL_41:
    sub_390E9A0(a1 + 96, 2 * v18);
    v28 = *(_DWORD *)(a1 + 120);
    if ( v28 )
    {
      v20 = v44;
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 104);
      v31 = v29 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v32 = (unsigned int)(*(_DWORD *)(a1 + 112) + 1);
      v25 = (_QWORD *)(v30 + 88LL * v31);
      v33 = *v25;
      if ( *v25 != v44 )
      {
        LODWORD(v19) = 1;
        v40 = 0;
        while ( v33 != -8 )
        {
          if ( v33 == -16 && !v40 )
            v40 = v25;
          v31 = v29 & (v19 + v31);
          v25 = (_QWORD *)(v30 + 88LL * v31);
          v33 = *v25;
          if ( v44 == *v25 )
            goto LABEL_43;
          LODWORD(v19) = v19 + 1;
        }
        if ( v40 )
          v25 = v40;
      }
      goto LABEL_43;
    }
    goto LABEL_80;
  }
LABEL_23:
  v11 = (__int64)(v25 + 1);
LABEL_24:
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  return v11;
}
