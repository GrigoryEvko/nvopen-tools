// Function: sub_B11390
// Address: 0xb11390
//
_QWORD *__fastcall sub_B11390(_QWORD *a1, __int64 a2, unsigned __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // esi
  __int64 *v14; // rcx
  __int64 v15; // r9
  int v16; // esi
  unsigned __int8 v17; // al
  __int64 v18; // rbx
  unsigned __int8 v19; // al
  unsigned __int8 **v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // r8
  __int64 v25; // rbx
  _BYTE *v26; // r14
  __int64 *v27; // rdx
  _QWORD *v28; // rax
  unsigned int v29; // esi
  __int64 v30; // r9
  int v31; // r11d
  _QWORD *v32; // rdx
  unsigned int v33; // edi
  _QWORD *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rbx
  unsigned __int8 v37; // al
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  int v42; // esi
  int v43; // esi
  __int64 v44; // r9
  unsigned int v45; // ecx
  __int64 v46; // rdi
  int v47; // r15d
  _QWORD *v48; // r10
  int v49; // ecx
  int v50; // r10d
  int v51; // ecx
  int v52; // ecx
  __int64 v53; // rdi
  _QWORD *v54; // r9
  unsigned int v55; // r15d
  int v56; // r11d
  __int64 v57; // rsi
  __int64 v58; // [rsp+0h] [rbp-A0h]
  __int64 v59; // [rsp+0h] [rbp-A0h]
  _QWORD *v61; // [rsp+18h] [rbp-88h]
  _BYTE *v62; // [rsp+18h] [rbp-88h]
  __int64 v63; // [rsp+28h] [rbp-78h] BYREF
  _BYTE *v64; // [rsp+30h] [rbp-70h] BYREF
  __int64 v65; // [rsp+38h] [rbp-68h]
  _BYTE v66[96]; // [rsp+40h] [rbp-60h] BYREF

  v64 = v66;
  v65 = 0x600000000LL;
  v8 = sub_B10CD0(a2);
  v9 = 0;
  if ( v8 )
  {
    v10 = v8;
    while ( 1 )
    {
      v11 = *(unsigned int *)(a5 + 24);
      v12 = *(_QWORD *)(a5 + 8);
      if ( (_DWORD)v11 )
      {
        v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v14 = (__int64 *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v10 == *v14 )
        {
LABEL_5:
          if ( v14 != (__int64 *)(v12 + 16 * v11) )
          {
            v24 = v14[1];
            if ( v24 )
              goto LABEL_19;
            break;
          }
        }
        else
        {
          v49 = 1;
          while ( v15 != -4096 )
          {
            v50 = v49 + 1;
            v13 = (v11 - 1) & (v49 + v13);
            v14 = (__int64 *)(v12 + 16LL * v13);
            v15 = *v14;
            if ( v10 == *v14 )
              goto LABEL_5;
            v49 = v50;
          }
        }
      }
      if ( v9 + 1 > (unsigned __int64)HIDWORD(v65) )
      {
        sub_C8D5F0(&v64, v66, v9 + 1, 8);
        v9 = (unsigned int)v65;
      }
      *(_QWORD *)&v64[8 * v9] = v10;
      v16 = v65;
      v9 = (unsigned int)(v65 + 1);
      LODWORD(v65) = v65 + 1;
      v17 = *(_BYTE *)(v10 - 16);
      if ( (v17 & 2) != 0 )
      {
        if ( *(_DWORD *)(v10 - 24) != 2 )
          goto LABEL_10;
        v25 = *(_QWORD *)(v10 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v10 - 16) >> 6) & 0xF) != 2 )
          goto LABEL_10;
        v25 = v10 - 16 - 8LL * ((v17 >> 2) & 0xF);
      }
      v10 = *(_QWORD *)(v25 + 8);
      if ( !v10 )
        goto LABEL_10;
    }
  }
  v16 = v9 - 1;
LABEL_10:
  v18 = *(_QWORD *)&v64[8 * v9 - 8];
  LODWORD(v65) = v16;
  v19 = *(_BYTE *)(v18 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(unsigned __int8 ***)(v18 - 32);
  else
    v20 = (unsigned __int8 **)(v18 - 16 - 8LL * ((v19 >> 2) & 0xF));
  v21 = sub_B00540(*v20, a3, (__int64)a4, a5);
  v22 = sub_B01860(a4, *(_DWORD *)(v18 + 4), *(unsigned __int16 *)(v18 + 2), v21, 0, 0, 0, 1);
  v63 = v18;
  v61 = v22;
  v23 = sub_B11140(a5, &v63);
  v24 = (__int64)v61;
  *v23 = v61;
  v9 = (unsigned int)v65;
LABEL_19:
  v26 = &v64[8 * v9];
  v62 = v64;
  if ( v64 != v26 )
  {
    while ( 1 )
    {
      v36 = *((_QWORD *)v26 - 1);
      v37 = *(_BYTE *)(v36 - 16);
      v27 = (v37 & 2) != 0 ? *(__int64 **)(v36 - 32) : (__int64 *)(v36 - 16 - 8LL * ((v37 >> 2) & 0xF));
      v28 = sub_B01860(a4, *(_DWORD *)(v36 + 4), *(unsigned __int16 *)(v36 + 2), *v27, v24, 0, 0, 1);
      v29 = *(_DWORD *)(a5 + 24);
      v24 = (__int64)v28;
      if ( !v29 )
        break;
      v30 = *(_QWORD *)(a5 + 8);
      v31 = 1;
      v32 = 0;
      v33 = (v29 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v34 = (_QWORD *)(v30 + 16LL * v33);
      v35 = *v34;
      if ( v36 == *v34 )
      {
LABEL_24:
        v26 -= 8;
        v34[1] = v24;
        if ( v62 == v26 )
          goto LABEL_40;
      }
      else
      {
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v32 )
            v32 = v34;
          v33 = (v29 - 1) & (v31 + v33);
          v34 = (_QWORD *)(v30 + 16LL * v33);
          v35 = *v34;
          if ( v36 == *v34 )
            goto LABEL_24;
          ++v31;
        }
        if ( !v32 )
          v32 = v34;
        v38 = *(_DWORD *)(a5 + 16);
        ++*(_QWORD *)a5;
        v39 = v38 + 1;
        if ( 4 * v39 < 3 * v29 )
        {
          if ( v29 - *(_DWORD *)(a5 + 20) - v39 <= v29 >> 3 )
          {
            v59 = v24;
            sub_B00360(a5, v29);
            v51 = *(_DWORD *)(a5 + 24);
            if ( !v51 )
            {
LABEL_73:
              ++*(_DWORD *)(a5 + 16);
              BUG();
            }
            v52 = v51 - 1;
            v53 = *(_QWORD *)(a5 + 8);
            v54 = 0;
            v55 = v52 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
            v24 = v59;
            v56 = 1;
            v39 = *(_DWORD *)(a5 + 16) + 1;
            v32 = (_QWORD *)(v53 + 16LL * v55);
            v57 = *v32;
            if ( v36 != *v32 )
            {
              while ( v57 != -4096 )
              {
                if ( v57 == -8192 && !v54 )
                  v54 = v32;
                v55 = v52 & (v56 + v55);
                v32 = (_QWORD *)(v53 + 16LL * v55);
                v57 = *v32;
                if ( v36 == *v32 )
                  goto LABEL_37;
                ++v56;
              }
              if ( v54 )
                v32 = v54;
            }
          }
          goto LABEL_37;
        }
LABEL_44:
        v58 = v24;
        sub_B00360(a5, 2 * v29);
        v42 = *(_DWORD *)(a5 + 24);
        if ( !v42 )
          goto LABEL_73;
        v43 = v42 - 1;
        v44 = *(_QWORD *)(a5 + 8);
        v24 = v58;
        v45 = v43 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v39 = *(_DWORD *)(a5 + 16) + 1;
        v32 = (_QWORD *)(v44 + 16LL * v45);
        v46 = *v32;
        if ( v36 != *v32 )
        {
          v47 = 1;
          v48 = 0;
          while ( v46 != -4096 )
          {
            if ( !v48 && v46 == -8192 )
              v48 = v32;
            v45 = v43 & (v47 + v45);
            v32 = (_QWORD *)(v44 + 16LL * v45);
            v46 = *v32;
            if ( v36 == *v32 )
              goto LABEL_37;
            ++v47;
          }
          if ( v48 )
            v32 = v48;
        }
LABEL_37:
        *(_DWORD *)(a5 + 16) = v39;
        if ( *v32 != -4096 )
          --*(_DWORD *)(a5 + 20);
        *v32 = v36;
        v26 -= 8;
        v32[1] = 0;
        v32[1] = v24;
        if ( v62 == v26 )
          goto LABEL_40;
      }
    }
    ++*(_QWORD *)a5;
    goto LABEL_44;
  }
LABEL_40:
  v40 = v24;
  sub_B10CB0(a1, v24);
  if ( v64 != v66 )
    _libc_free(v64, v40);
  return a1;
}
