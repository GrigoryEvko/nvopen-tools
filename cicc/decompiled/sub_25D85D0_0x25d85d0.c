// Function: sub_25D85D0
// Address: 0x25d85d0
//
void __fastcall sub_25D85D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 *v9; // rax
  __int64 *v10; // r13
  __int64 *v11; // rdx
  __int64 v12; // r12
  __int64 *v13; // rbx
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  unsigned int v16; // esi
  int v17; // r11d
  __int64 v18; // r9
  __int64 i; // rdx
  __int64 v20; // r8
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 *v25; // rax
  int v26; // ecx
  int v27; // ecx
  _QWORD *v28; // rax
  __int64 *v29; // rax
  int v30; // esi
  int v31; // esi
  __int64 v32; // rdx
  __int64 v33; // rdi
  int v34; // r11d
  int v35; // esi
  int v36; // esi
  int v37; // r11d
  __int64 v38; // rdx
  __int64 v39; // rdi
  unsigned int v40; // [rsp+Ch] [rbp-A4h]
  __int64 v41; // [rsp+10h] [rbp-A0h]
  __int64 v42; // [rsp+18h] [rbp-98h]
  __int64 v43; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v44; // [rsp+28h] [rbp-88h]
  __int64 v45; // [rsp+30h] [rbp-80h]
  int v46; // [rsp+38h] [rbp-78h]
  char v47; // [rsp+3Ch] [rbp-74h]
  char v48; // [rsp+40h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a2 + 16);
  v43 = 0;
  v44 = (__int64 *)&v48;
  v45 = 8;
  v46 = 0;
  v47 = 1;
  if ( !v6 )
    goto LABEL_14;
  do
  {
    sub_25D7390((_QWORD *)a1, *(unsigned __int8 **)(v6 + 24), (__int64)&v43, a4, a5, a6);
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v6 );
  if ( v47 )
  {
    v9 = v44;
    v10 = &v44[HIDWORD(v45)];
    if ( v10 == v44 )
      goto LABEL_14;
    v11 = v44;
    while ( a2 != *v11 )
    {
      if ( ++v11 == v10 )
        goto LABEL_12;
    }
    --HIDWORD(v45);
    *v11 = v44[HIDWORD(v45)];
    ++v43;
  }
  else
  {
    v29 = sub_C8CA60((__int64)&v43, a2);
    if ( v29 )
    {
      *v29 = -2;
      v9 = v44;
      ++v46;
      ++v43;
      if ( !v47 )
        goto LABEL_10;
      goto LABEL_55;
    }
  }
  v9 = v44;
  if ( !v47 )
  {
LABEL_10:
    v10 = &v9[(unsigned int)v45];
    goto LABEL_11;
  }
LABEL_55:
  v10 = &v9[HIDWORD(v45)];
LABEL_11:
  if ( v9 != v10 )
  {
LABEL_12:
    while ( 1 )
    {
      v12 = *v9;
      v13 = v9;
      if ( (unsigned __int64)*v9 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v9 )
        goto LABEL_14;
    }
    if ( v10 != v9 )
    {
      v42 = a1 + 472;
      v41 = a1 + 296;
      if ( !*(_BYTE *)(a1 + 500) )
        goto LABEL_33;
LABEL_18:
      v14 = *(_QWORD **)(a1 + 480);
      v15 = &v14[*(unsigned int *)(a1 + 492)];
      if ( v14 == v15 )
      {
LABEL_23:
        v16 = *(_DWORD *)(a1 + 320);
        if ( v16 )
        {
          v17 = 1;
          v18 = *(_QWORD *)(a1 + 304);
          i = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
          v20 = (v16 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v21 = v18 + 72 * v20;
          v22 = 0;
          v23 = *(_QWORD *)v21;
          if ( v12 == *(_QWORD *)v21 )
          {
LABEL_25:
            v24 = v21 + 8;
            if ( !*(_BYTE *)(v21 + 36) )
              goto LABEL_26;
            goto LABEL_48;
          }
          while ( v23 != -4096 )
          {
            if ( !v22 && v23 == -8192 )
              v22 = v21;
            v20 = (v16 - 1) & (v17 + (_DWORD)v20);
            v21 = v18 + 72LL * (unsigned int)v20;
            v23 = *(_QWORD *)v21;
            if ( v12 == *(_QWORD *)v21 )
              goto LABEL_25;
            ++v17;
          }
          if ( !v22 )
            v22 = v21;
          v26 = *(_DWORD *)(a1 + 312);
          ++*(_QWORD *)(a1 + 296);
          v27 = v26 + 1;
          if ( 4 * v27 < 3 * v16 )
          {
            v20 = v16 >> 3;
            if ( v16 - *(_DWORD *)(a1 + 316) - v27 > (unsigned int)v20 )
            {
LABEL_45:
              *(_DWORD *)(a1 + 312) = v27;
              if ( *(_QWORD *)v22 != -4096 )
                --*(_DWORD *)(a1 + 316);
              *(_QWORD *)v22 = v12;
              v24 = v22 + 8;
              *(_QWORD *)(v22 + 8) = 0;
              *(_QWORD *)(v22 + 16) = v22 + 40;
              *(_QWORD *)(v22 + 24) = 4;
              *(_DWORD *)(v22 + 32) = 0;
              *(_BYTE *)(v22 + 36) = 1;
LABEL_48:
              v28 = *(_QWORD **)(v24 + 8);
              v21 = *(unsigned int *)(v24 + 20);
              for ( i = (__int64)&v28[v21]; (_QWORD *)i != v28; ++v28 )
              {
                if ( a2 == *v28 )
                  goto LABEL_27;
              }
              if ( (unsigned int)v21 < *(_DWORD *)(v24 + 16) )
              {
                *(_DWORD *)(v24 + 20) = v21 + 1;
                *(_QWORD *)i = a2;
                ++*(_QWORD *)v24;
                goto LABEL_27;
              }
LABEL_26:
              sub_C8CC70(v24, a2, i, v21, v20, v18);
              goto LABEL_27;
            }
            v40 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
            sub_25D83A0(v41, v16);
            v35 = *(_DWORD *)(a1 + 320);
            if ( v35 )
            {
              v36 = v35 - 1;
              v20 = 0;
              v18 = *(_QWORD *)(a1 + 304);
              v37 = 1;
              LODWORD(v38) = v36 & v40;
              v27 = *(_DWORD *)(a1 + 312) + 1;
              v22 = v18 + 72LL * (v36 & v40);
              v39 = *(_QWORD *)v22;
              if ( v12 == *(_QWORD *)v22 )
                goto LABEL_45;
              while ( v39 != -4096 )
              {
                if ( v39 == -8192 && !v20 )
                  v20 = v22;
                v38 = v36 & (unsigned int)(v38 + v37);
                v22 = v18 + 72 * v38;
                v39 = *(_QWORD *)v22;
                if ( v12 == *(_QWORD *)v22 )
                  goto LABEL_45;
                ++v37;
              }
              goto LABEL_70;
            }
            goto LABEL_78;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 296);
        }
        sub_25D83A0(v41, 2 * v16);
        v30 = *(_DWORD *)(a1 + 320);
        if ( v30 )
        {
          v31 = v30 - 1;
          v18 = *(_QWORD *)(a1 + 304);
          LODWORD(v32) = v31 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v27 = *(_DWORD *)(a1 + 312) + 1;
          v22 = v18 + 72LL * (unsigned int)v32;
          v33 = *(_QWORD *)v22;
          if ( v12 == *(_QWORD *)v22 )
            goto LABEL_45;
          v20 = 0;
          v34 = 1;
          while ( v33 != -4096 )
          {
            if ( v33 == -8192 && !v20 )
              v20 = v22;
            v32 = v31 & (unsigned int)(v32 + v34);
            v22 = v18 + 72 * v32;
            v33 = *(_QWORD *)v22;
            if ( v12 == *(_QWORD *)v22 )
              goto LABEL_45;
            ++v34;
          }
LABEL_70:
          if ( v20 )
            v22 = v20;
          goto LABEL_45;
        }
LABEL_78:
        ++*(_DWORD *)(a1 + 312);
        BUG();
      }
      while ( v12 != *v14 )
      {
        if ( v15 == ++v14 )
          goto LABEL_23;
      }
      while ( 1 )
      {
        if ( *(_BYTE *)a2 )
          goto LABEL_23;
LABEL_27:
        v25 = v13 + 1;
        if ( v13 + 1 == v10 )
          goto LABEL_14;
        v12 = *v25;
        ++v13;
        if ( (unsigned __int64)*v25 >= 0xFFFFFFFFFFFFFFFELL )
          break;
LABEL_31:
        if ( v13 == v10 )
          goto LABEL_14;
        if ( *(_BYTE *)(a1 + 500) )
          goto LABEL_18;
LABEL_33:
        if ( !sub_C8CA60(v42, v12) )
          goto LABEL_23;
      }
      while ( v10 != ++v25 )
      {
        v12 = *v25;
        v13 = v25;
        if ( (unsigned __int64)*v25 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_31;
      }
    }
  }
LABEL_14:
  if ( !v47 )
    _libc_free((unsigned __int64)v44);
}
