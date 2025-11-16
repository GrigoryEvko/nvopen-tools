// Function: sub_27468F0
// Address: 0x27468f0
//
void __fastcall sub_27468F0(__int64 *a1, const void *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v6; // r15
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r8
  _BYTE *v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // r15
  __int64 v15; // rdx
  unsigned __int8 *v16; // rbx
  __int64 v17; // rdi
  unsigned __int8 **v18; // rax
  char v19; // dl
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int8 *v24; // r10
  int v25; // edx
  __int64 v26; // rsi
  int v27; // ecx
  unsigned __int8 *v28; // r10
  int v29; // ecx
  unsigned int v30; // eax
  unsigned __int8 *v31; // r9
  _QWORD *v32; // rdi
  unsigned __int8 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // edx
  int v37; // r11d
  int v38; // r11d
  unsigned __int64 v39; // rdx
  __int64 v40; // rax
  char *v41; // rdx
  char *v42; // rcx
  unsigned __int64 v43; // [rsp+0h] [rbp-70h]
  __int64 v44; // [rsp+0h] [rbp-70h]
  __int64 v45; // [rsp+0h] [rbp-70h]
  _BYTE *v46; // [rsp+10h] [rbp-60h] BYREF
  __int64 v47; // [rsp+18h] [rbp-58h]
  _BYTE v48[80]; // [rsp+20h] [rbp-50h] BYREF

  v6 = 8 * a3;
  v8 = *a1;
  v46 = v48;
  v9 = v8 + 1232;
  v10 = v8 + 600;
  if ( !(_BYTE)a4 )
    v9 = v10;
  v11 = (8 * a3) >> 3;
  v47 = 0x400000000LL;
  if ( v6 > 0x20 )
  {
    v43 = (8 * a3) >> 3;
    sub_C8D5F0((__int64)&v46, v48, v43, 8u, v11, a6);
    v11 = v43;
    v12 = &v46[8 * (unsigned int)v47];
  }
  else
  {
    v12 = v48;
    if ( !v6 )
      goto LABEL_5;
  }
  v44 = v11;
  memcpy(v12, a2, v6);
  LODWORD(v6) = v47;
  v12 = v46;
  v11 = v44;
LABEL_5:
  v13 = v11 + v6;
  LODWORD(v47) = v13;
  v14 = 0x33FFE23FFFFLL;
  if ( v13 )
  {
    while ( 1 )
    {
      v15 = v13;
      v16 = *(unsigned __int8 **)&v12[8 * v13 - 8];
      v17 = a1[1];
      LODWORD(v47) = v13 - 1;
      if ( !*(_BYTE *)(v17 + 28) )
        goto LABEL_18;
      v18 = *(unsigned __int8 ***)(v17 + 8);
      a4 = *(unsigned int *)(v17 + 20);
      v15 = (__int64)&v18[a4];
      if ( v18 != (unsigned __int8 **)v15 )
      {
        while ( v16 != *v18 )
        {
          if ( (unsigned __int8 **)v15 == ++v18 )
            goto LABEL_36;
        }
LABEL_11:
        v12 = v46;
        v13 = v47;
        goto LABEL_12;
      }
LABEL_36:
      if ( (unsigned int)a4 < *(_DWORD *)(v17 + 16) )
      {
        a4 = (unsigned int)(a4 + 1);
        *(_DWORD *)(v17 + 20) = a4;
        *(_QWORD *)v15 = v16;
        ++*(_QWORD *)v17;
      }
      else
      {
LABEL_18:
        sub_C8CC70(v17, (__int64)v16, v15, a4, v11, a6);
        if ( !v19 )
          goto LABEL_11;
      }
      v20 = a1[2];
      v21 = *(unsigned int *)(v20 + 24);
      if ( (_DWORD)v21 )
      {
        a6 = (unsigned int)(v21 - 1);
        v22 = *(_QWORD *)(v20 + 8);
        a4 = (unsigned int)a6 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v23 = v22 + (a4 << 6);
        v24 = *(unsigned __int8 **)(v23 + 24);
        if ( v16 == v24 )
        {
LABEL_21:
          if ( v23 != v22 + (v21 << 6) )
            goto LABEL_11;
        }
        else
        {
          v36 = 1;
          while ( v24 != (unsigned __int8 *)-4096LL )
          {
            v37 = v36 + 1;
            a4 = (unsigned int)a6 & (v36 + (_DWORD)a4);
            v23 = v22 + ((unsigned __int64)(unsigned int)a4 << 6);
            v24 = *(unsigned __int8 **)(v23 + 24);
            if ( v16 == v24 )
              goto LABEL_21;
            v36 = v37;
          }
        }
      }
      v25 = *v16;
      if ( (unsigned __int8)v25 <= 0x15u )
        goto LABEL_11;
      v26 = *(_QWORD *)(v9 + 8);
      v27 = *(_DWORD *)(v9 + 24);
      if ( (unsigned __int8)v25 <= 0x1Cu )
        break;
      v28 = v16;
      if ( v27 )
        goto LABEL_25;
LABEL_48:
      v39 = (unsigned int)(v25 - 42);
      if ( (unsigned __int8)v39 > 0x29u || !_bittest64(&v14, v39) )
        goto LABEL_26;
      v40 = 32LL * (*((_DWORD *)v28 + 1) & 0x7FFFFFF);
      if ( (v28[7] & 0x40) != 0 )
      {
        v41 = (char *)*((_QWORD *)v28 - 1);
        v42 = &v41[v40];
      }
      else
      {
        v42 = (char *)v28;
        v41 = (char *)&v28[-v40];
      }
      sub_2739020((__int64)&v46, &v46[8 * (unsigned int)v47], v41, v42);
      v12 = v46;
      v13 = v47;
LABEL_12:
      if ( !v13 )
        goto LABEL_13;
    }
    if ( !v27 )
      goto LABEL_26;
    v28 = 0;
LABEL_25:
    v29 = v27 - 1;
    v30 = v29 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v31 = *(unsigned __int8 **)(v26 + 16LL * v30);
    if ( v16 == v31 )
      goto LABEL_26;
    v38 = 1;
    while ( v31 != (unsigned __int8 *)-4096LL )
    {
      v30 = v29 & (v38 + v30);
      v31 = *(unsigned __int8 **)(v26 + 16LL * v30);
      if ( v16 == v31 )
        goto LABEL_26;
      ++v38;
    }
    if ( !v28 || (unsigned __int8)v25 <= 0x1Cu )
    {
LABEL_26:
      v32 = sub_2745840(v20, (__int64)v16);
      v33 = (unsigned __int8 *)v32[2];
      if ( v16 != v33 )
      {
        if ( v33 + 4096 != 0 && v33 != 0 && v33 != (unsigned __int8 *)-8192LL )
          sub_BD60C0(v32);
        v32[2] = v16;
        if ( v16 != (unsigned __int8 *)-8192LL && v16 != (unsigned __int8 *)-4096LL )
          sub_BD73F0((__int64)v32);
      }
      v34 = a1[3];
      v35 = *(unsigned int *)(v34 + 8);
      a6 = v35 + 1;
      if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v34 + 12) )
      {
        v45 = a1[3];
        sub_C8D5F0(v45, (const void *)(v34 + 16), v35 + 1, 8u, v11, a6);
        v34 = v45;
        v35 = *(unsigned int *)(v45 + 8);
      }
      a4 = *(_QWORD *)v34;
      *(_QWORD *)(*(_QWORD *)v34 + 8 * v35) = v16;
      v12 = v46;
      ++*(_DWORD *)(v34 + 8);
      v13 = v47;
      goto LABEL_12;
    }
    goto LABEL_48;
  }
LABEL_13:
  if ( v12 != v48 )
    _libc_free((unsigned __int64)v12);
}
