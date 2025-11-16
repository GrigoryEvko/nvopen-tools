// Function: sub_2C60650
// Address: 0x2c60650
//
void __fastcall sub_2C60650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  _QWORD *v8; // r12
  _QWORD *v9; // r13
  _QWORD *v10; // rax
  int v11; // ecx
  __int64 v12; // r15
  _BYTE *v13; // rdi
  _BYTE **v14; // r11
  _BYTE *v15; // r10
  __int64 v16; // r12
  __int64 v17; // r9
  int v18; // r10d
  __int64 v19; // r8
  _QWORD *v20; // rdx
  unsigned int v21; // edi
  _QWORD *v22; // rax
  __int64 v23; // rcx
  unsigned int v24; // esi
  __int64 v25; // r13
  int v26; // r15d
  int v27; // esi
  int v28; // esi
  unsigned int v29; // ecx
  int v30; // eax
  __int64 v31; // rdi
  __int64 v32; // rax
  int v33; // eax
  int v34; // ecx
  int v35; // ecx
  __int64 v36; // rdi
  __int64 v37; // r14
  int v38; // r10d
  __int64 v39; // rsi
  int v40; // r11d
  __int64 v41; // [rsp+18h] [rbp-98h]
  __int64 v42; // [rsp+20h] [rbp-90h]
  _BYTE **v43; // [rsp+28h] [rbp-88h]
  _BYTE **v44; // [rsp+30h] [rbp-80h]
  _BYTE *v45; // [rsp+38h] [rbp-78h]
  _BYTE **v46; // [rsp+38h] [rbp-78h]
  _BYTE *v47; // [rsp+40h] [rbp-70h] BYREF
  __int64 v48; // [rsp+48h] [rbp-68h]
  _BYTE v49[96]; // [rsp+50h] [rbp-60h] BYREF

  v7 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v8 = *(_QWORD **)(a2 - 8);
    v9 = &v8[(unsigned __int64)v7 / 8];
  }
  else
  {
    v9 = (_QWORD *)a2;
    v8 = (_QWORD *)(a2 - v7);
  }
  v10 = v49;
  v11 = 0;
  v12 = v7 >> 5;
  v47 = v49;
  v48 = 0x600000000LL;
  if ( (unsigned __int64)v7 > 0xC0 )
  {
    sub_C8D5F0((__int64)&v47, v49, v7 >> 5, 8u, a5, a6);
    v11 = v48;
    v10 = &v47[8 * (unsigned int)v48];
  }
  if ( v8 != v9 )
  {
    do
    {
      if ( v10 )
        *v10 = *v8;
      v8 += 4;
      ++v10;
    }
    while ( v8 != v9 );
    v11 = v48;
  }
  v42 = a1 + 200;
  LODWORD(v48) = v11 + v12;
  sub_F0BA50(a1 + 200, a2);
  sub_B43D60((_QWORD *)a2);
  v13 = v47;
  v43 = (_BYTE **)&v47[8 * (unsigned int)v48];
  if ( v43 != (_BYTE **)v47 )
  {
    v14 = (_BYTE **)v47;
    v41 = a1 + 2264;
    while ( 1 )
    {
      v15 = *v14;
      if ( **v14 <= 0x1Cu )
        goto LABEL_28;
      v16 = *((_QWORD *)v15 + 2);
      if ( !v16 )
        goto LABEL_27;
      v45 = *v14;
      v44 = v14;
      do
      {
        while ( 1 )
        {
          v24 = *(_DWORD *)(a1 + 2288);
          v25 = *(_QWORD *)(v16 + 24);
          v26 = *(_DWORD *)(a1 + 208);
          if ( !v24 )
          {
            ++*(_QWORD *)(a1 + 2264);
LABEL_19:
            sub_9BAAD0(v41, 2 * v24);
            v27 = *(_DWORD *)(a1 + 2288);
            if ( !v27 )
              goto LABEL_65;
            v28 = v27 - 1;
            v19 = *(_QWORD *)(a1 + 2272);
            v29 = v28 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v30 = *(_DWORD *)(a1 + 2280) + 1;
            v20 = (_QWORD *)(v19 + 16LL * v29);
            v31 = *v20;
            if ( v25 != *v20 )
            {
              v40 = 1;
              v17 = 0;
              while ( v31 != -4096 )
              {
                if ( v31 == -8192 && !v17 )
                  v17 = (__int64)v20;
                v29 = v28 & (v40 + v29);
                v20 = (_QWORD *)(v19 + 16LL * v29);
                v31 = *v20;
                if ( v25 == *v20 )
                  goto LABEL_21;
                ++v40;
              }
              if ( v17 )
                v20 = (_QWORD *)v17;
            }
            goto LABEL_21;
          }
          v17 = v24 - 1;
          v18 = 1;
          v19 = *(_QWORD *)(a1 + 2272);
          v20 = 0;
          v21 = v17 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v22 = (_QWORD *)(v19 + 16LL * v21);
          v23 = *v22;
          if ( v25 != *v22 )
            break;
LABEL_16:
          v16 = *(_QWORD *)(v16 + 8);
          if ( !v16 )
            goto LABEL_26;
        }
        while ( v23 != -4096 )
        {
          if ( v20 || v23 != -8192 )
            v22 = v20;
          v21 = v17 & (v18 + v21);
          v23 = *(_QWORD *)(v19 + 16LL * v21);
          if ( v25 == v23 )
            goto LABEL_16;
          ++v18;
          v20 = v22;
          v22 = (_QWORD *)(v19 + 16LL * v21);
        }
        if ( !v20 )
          v20 = v22;
        v33 = *(_DWORD *)(a1 + 2280);
        ++*(_QWORD *)(a1 + 2264);
        v30 = v33 + 1;
        if ( 4 * v30 >= 3 * v24 )
          goto LABEL_19;
        if ( v24 - *(_DWORD *)(a1 + 2284) - v30 <= v24 >> 3 )
        {
          sub_9BAAD0(v41, v24);
          v34 = *(_DWORD *)(a1 + 2288);
          if ( !v34 )
          {
LABEL_65:
            ++*(_DWORD *)(a1 + 2280);
            BUG();
          }
          v35 = v34 - 1;
          v36 = *(_QWORD *)(a1 + 2272);
          v19 = 0;
          LODWORD(v37) = v35 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v38 = 1;
          v30 = *(_DWORD *)(a1 + 2280) + 1;
          v20 = (_QWORD *)(v36 + 16LL * (unsigned int)v37);
          v39 = *v20;
          if ( v25 != *v20 )
          {
            while ( v39 != -4096 )
            {
              if ( !v19 && v39 == -8192 )
                v19 = (__int64)v20;
              v17 = (unsigned int)(v38 + 1);
              v37 = v35 & (unsigned int)(v37 + v38);
              v20 = (_QWORD *)(v36 + 16 * v37);
              v39 = *v20;
              if ( v25 == *v20 )
                goto LABEL_21;
              ++v38;
            }
            if ( v19 )
              v20 = (_QWORD *)v19;
          }
        }
LABEL_21:
        *(_DWORD *)(a1 + 2280) = v30;
        if ( *v20 != -4096 )
          --*(_DWORD *)(a1 + 2284);
        *v20 = v25;
        *((_DWORD *)v20 + 2) = v26;
        v32 = *(unsigned int *)(a1 + 208);
        if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 212) )
        {
          sub_C8D5F0(v42, (const void *)(a1 + 216), v32 + 1, 8u, v19, v17);
          v32 = *(unsigned int *)(a1 + 208);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v32) = v25;
        ++*(_DWORD *)(a1 + 208);
        v16 = *(_QWORD *)(v16 + 8);
      }
      while ( v16 );
LABEL_26:
      v15 = v45;
      v14 = v44;
      if ( *v45 > 0x1Cu )
      {
LABEL_27:
        v46 = v14;
        sub_F15FC0(v42, (__int64)v15);
        v14 = v46;
      }
LABEL_28:
      if ( v43 == ++v14 )
      {
        v13 = v47;
        break;
      }
    }
  }
  if ( v13 != v49 )
    _libc_free((unsigned __int64)v13);
}
