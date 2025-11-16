// Function: sub_E496B0
// Address: 0xe496b0
//
__int64 __fastcall sub_E496B0(__int64 a1, __int64 a2, __int64 a3)
{
  int v6; // eax
  __int64 v7; // r14
  const char *v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // r14
  unsigned int v13; // r13d
  int v15; // eax
  unsigned int v16; // eax
  __int64 v17; // rax
  unsigned int v18; // esi
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r8
  unsigned int v23; // eax
  char v24; // al
  __int64 v25; // rax
  unsigned __int8 v26; // dl
  unsigned __int8 v27; // al
  unsigned __int8 v28; // cl
  int v29; // eax
  int v30; // edx
  char v31; // al
  char v32; // al
  char v33; // al
  int v34; // edx
  int v35; // eax
  __int64 *v36; // rdi
  int v37; // ebx
  int v38; // r8d
  int v39; // edx
  int v40; // ecx
  __int64 v41; // r9
  unsigned int v42; // edx
  __int64 v43; // rsi
  int v44; // r11d
  __int64 *v45; // r10
  int v46; // edx
  int v47; // edx
  int v48; // r11d
  __int64 v49; // r9
  unsigned int v50; // ecx
  __int64 v51; // rsi
  unsigned __int16 v52; // ax
  __int16 v53; // dx
  unsigned __int8 v54; // dl
  unsigned __int16 v55; // r13
  __int16 v56; // dx
  __int64 v57; // [rsp+8h] [rbp-58h]
  int v58; // [rsp+14h] [rbp-4Ch]
  unsigned int v59; // [rsp+14h] [rbp-4Ch]
  unsigned int v60; // [rsp+18h] [rbp-48h]
  unsigned int v61; // [rsp+18h] [rbp-48h]
  __int64 v62; // [rsp+18h] [rbp-48h]
  __int64 v63; // [rsp+18h] [rbp-48h]
  char v64; // [rsp+27h] [rbp-39h] BYREF
  __int64 v65[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v6 = *(_BYTE *)(a2 + 32) & 0xF;
    if ( v6 == 7 || v6 == 8 )
    {
      v15 = *(_DWORD *)(a1 + 64);
      if ( (v15 & 2) != 0 || (v15 & 1) == 0 )
        return 0;
      goto LABEL_12;
    }
    v7 = **(_QWORD **)a1;
    v8 = sub_BD5D20(a2);
    v10 = sub_BA8B30(v7, (__int64)v8, v9);
    v11 = *(_DWORD *)(a1 + 64);
    v12 = v10;
    if ( !v10 || (*(_BYTE *)(v10 + 32) & 0xFu) - 7 <= 1 )
    {
      if ( (v11 & 2) != 0 )
        goto LABEL_7;
LABEL_30:
      if ( (v11 & 1) == 0 && ((*(_BYTE *)(a2 + 32) & 0xFu) - 7 <= 1 || (((*(_BYTE *)(a2 + 32) & 0xF) + 15) & 0xFu) <= 2) )
        return 0;
      goto LABEL_12;
    }
    if ( (v11 & 2) != 0 )
    {
      if ( (*(_BYTE *)(a2 + 32) & 0xF) == 6 )
        goto LABEL_13;
      if ( !sub_B2FC80(v10) )
        return 0;
    }
    v26 = *(_BYTE *)(a2 + 32);
    if ( ((v26 + 10) & 0xFu) <= 2 )
    {
LABEL_13:
      LOBYTE(v16) = sub_B2FC80(a2);
      v13 = v16;
      if ( (_BYTE)v16 )
        return 0;
      v17 = sub_B326A0(a2);
      if ( !v17 )
      {
        v64 = 1;
        if ( !v12 || (*(_BYTE *)(a1 + 64) & 1) != 0 )
          goto LABEL_29;
        v23 = sub_E48260(a1, (bool *)&v64, v12, a2);
        if ( !(_BYTE)v23 )
          goto LABEL_27;
        return v23;
      }
      v18 = *(_DWORD *)(a1 + 152);
      if ( v18 )
      {
        v19 = *(_QWORD *)(a1 + 136);
        v60 = (v18 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v20 = v19 + 16LL * v60;
        v21 = *(_QWORD *)v20;
        if ( v17 == *(_QWORD *)v20 )
        {
LABEL_17:
          v22 = *(unsigned int *)(v20 + 12);
          if ( !(_DWORD)v22 )
            return 0;
          v64 = 1;
          if ( !v12 )
          {
LABEL_29:
            v65[0] = a2;
            sub_E49430(a1 + 16, v65);
            return v13;
          }
          if ( (*(_BYTE *)(a1 + 64) & 1) != 0 )
          {
            if ( (_DWORD)v22 != 2 )
              goto LABEL_29;
LABEL_24:
            v25 = *(unsigned int *)(a3 + 8);
            if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
            {
              sub_C8D5F0(a3, (const void *)(a3 + 16), v25 + 1, 8u, v22, v19);
              v25 = *(unsigned int *)(a3 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a3 + 8 * v25) = v12;
            ++*(_DWORD *)(a3 + 8);
LABEL_27:
            v24 = v64;
LABEL_28:
            if ( !v24 )
              return 0;
            goto LABEL_29;
          }
          v61 = v22;
          v23 = sub_E48260(a1, (bool *)&v64, v12, a2);
          v22 = v61;
          if ( !(_BYTE)v23 )
          {
            v24 = v64;
            if ( v61 != 2 )
              goto LABEL_28;
            if ( !v64 )
              v12 = a2;
            goto LABEL_24;
          }
          return v23;
        }
        v58 = 1;
        v36 = 0;
        v57 = v19;
        while ( v21 != -4096 )
        {
          if ( v21 == -8192 && !v36 )
            v36 = (__int64 *)v20;
          v19 = (unsigned int)(v58 + 1);
          v60 = (v18 - 1) & (v58 + v60);
          v20 = v57 + 16LL * v60;
          v21 = *(_QWORD *)v20;
          if ( v17 == *(_QWORD *)v20 )
            goto LABEL_17;
          ++v58;
        }
        v37 = *(_DWORD *)(a1 + 144);
        if ( !v36 )
          v36 = (__int64 *)v20;
        ++*(_QWORD *)(a1 + 128);
        v38 = v37 + 1;
        if ( 4 * (v37 + 1) < 3 * v18 )
        {
          if ( v18 - *(_DWORD *)(a1 + 148) - v38 > v18 >> 3 )
          {
LABEL_92:
            *(_DWORD *)(a1 + 144) = v38;
            if ( *v36 != -4096 )
              --*(_DWORD *)(a1 + 148);
            *v36 = v17;
            v36[1] = 0;
            return 0;
          }
          v59 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
          v63 = v17;
          sub_E47D80(a1 + 128, v18);
          v46 = *(_DWORD *)(a1 + 152);
          if ( v46 )
          {
            v47 = v46 - 1;
            v48 = 1;
            v45 = 0;
            v49 = *(_QWORD *)(a1 + 136);
            v50 = v47 & v59;
            v38 = *(_DWORD *)(a1 + 144) + 1;
            v17 = v63;
            v36 = (__int64 *)(v49 + 16LL * (v47 & v59));
            v51 = *v36;
            if ( v63 == *v36 )
              goto LABEL_92;
            while ( v51 != -4096 )
            {
              if ( !v45 && v51 == -8192 )
                v45 = v36;
              v50 = v47 & (v48 + v50);
              v36 = (__int64 *)(v49 + 16LL * v50);
              v51 = *v36;
              if ( v63 == *v36 )
                goto LABEL_92;
              ++v48;
            }
            goto LABEL_100;
          }
          goto LABEL_129;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 128);
      }
      v62 = v17;
      sub_E47D80(a1 + 128, 2 * v18);
      v39 = *(_DWORD *)(a1 + 152);
      if ( v39 )
      {
        v17 = v62;
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a1 + 136);
        v38 = *(_DWORD *)(a1 + 144) + 1;
        v42 = (v39 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v36 = (__int64 *)(v41 + 16LL * v42);
        v43 = *v36;
        if ( v62 == *v36 )
          goto LABEL_92;
        v44 = 1;
        v45 = 0;
        while ( v43 != -4096 )
        {
          if ( !v45 && v43 == -8192 )
            v45 = v36;
          v42 = v40 & (v44 + v42);
          v36 = (__int64 *)(v41 + 16LL * v42);
          v43 = *v36;
          if ( v62 == *v36 )
            goto LABEL_92;
          ++v44;
        }
LABEL_100:
        if ( v45 )
          v36 = v45;
        goto LABEL_92;
      }
LABEL_129:
      ++*(_DWORD *)(a1 + 144);
      BUG();
    }
    if ( *(_BYTE *)v12 != 3 || *(_BYTE *)a2 != 3 )
    {
      v27 = *(_BYTE *)(v12 + 32);
      v28 = v27 & 0xF;
      goto LABEL_46;
    }
    if ( sub_B2FC80(v12) && sub_B2FC80(a2) && ((*(_BYTE *)(v12 + 80) & 1) == 0 || (*(_BYTE *)(a2 + 80) & 1) == 0) )
    {
      *(_BYTE *)(v12 + 80) &= ~1u;
      *(_BYTE *)(a2 + 80) &= ~1u;
    }
    v27 = *(_BYTE *)(v12 + 32);
    v26 = *(_BYTE *)(a2 + 32);
    v28 = v27 & 0xF;
    if ( (v27 & 0xF) != 0xA )
    {
LABEL_46:
      v29 = (v27 >> 4) & 3;
      v30 = (v26 >> 4) & 3;
      if ( v29 == 1 || v30 == 1 )
      {
        v31 = 1;
      }
      else
      {
        if ( v29 != 2 && v30 != 2 )
        {
LABEL_50:
          v31 = 0;
          goto LABEL_51;
        }
        v31 = 2;
      }
LABEL_51:
      *(_BYTE *)(v12 + 32) = (16 * v31) | *(_BYTE *)(v12 + 32) & 0xCF;
      if ( (unsigned int)v28 - 7 <= 1 )
      {
LABEL_52:
        *(_BYTE *)(v12 + 33) |= 0x40u;
LABEL_53:
        v32 = (16 * v31) | *(_BYTE *)(a2 + 32) & 0xCF;
        *(_BYTE *)(a2 + 32) = v32;
        if ( (v32 & 0xFu) - 7 <= 1 || (v32 & 0x30) != 0 && (v32 & 0xF) != 9 )
          *(_BYTE *)(a2 + 33) |= 0x40u;
        if ( *(_BYTE *)(v12 + 32) >> 6 && *(_BYTE *)(a2 + 32) >> 6 )
        {
          if ( *(_BYTE *)(v12 + 32) >> 6 == 1 || (v33 = 2, *(_BYTE *)(a2 + 32) >> 6 == 1) )
            v33 = 1;
        }
        else
        {
          v33 = 0;
        }
        *(_BYTE *)(v12 + 32) = (v33 << 6) | *(_BYTE *)(v12 + 32) & 0x3F;
        *(_BYTE *)(a2 + 32) = (v33 << 6) | *(_BYTE *)(a2 + 32) & 0x3F;
        goto LABEL_13;
      }
LABEL_67:
      if ( (*(_BYTE *)(v12 + 32) & 0x30) == 0 || v28 == 9 )
        goto LABEL_53;
      goto LABEL_52;
    }
    if ( (v26 & 0xF) != 0xA )
    {
      v34 = (v26 >> 4) & 3;
      v35 = (v27 >> 4) & 3;
      if ( v34 == 1 || v35 == 1 )
      {
        *(_BYTE *)(v12 + 32) = *(_BYTE *)(v12 + 32) & 0xCF | 0x10;
        v31 = 1;
        goto LABEL_67;
      }
      if ( v35 == 2 || v34 == 2 )
      {
        *(_BYTE *)(v12 + 32) = *(_BYTE *)(v12 + 32) & 0xCF | 0x20;
        v31 = 2;
        goto LABEL_67;
      }
      goto LABEL_50;
    }
    v52 = (*(_WORD *)(v12 + 34) >> 1) & 0x3F;
    if ( v52 )
    {
      LOBYTE(v52) = v52 - 1;
      v53 = (*(_WORD *)(a2 + 34) >> 1) & 0x3F;
      if ( v53 )
      {
        v54 = v53 - 1;
        if ( (unsigned __int8)v52 < v54 )
          LOBYTE(v52) = v54;
      }
    }
    else
    {
      v55 = 0;
      v56 = (*(_WORD *)(a2 + 34) >> 1) & 0x3F;
      if ( !v56 )
        goto LABEL_116;
      LOBYTE(v52) = v56 - 1;
    }
    HIBYTE(v52) = 1;
    v55 = v52;
LABEL_116:
    sub_B2F740(a2, v55);
    sub_B2F740(v12, v55);
    v27 = *(_BYTE *)(v12 + 32);
    v26 = *(_BYTE *)(a2 + 32);
    v28 = v27 & 0xF;
    goto LABEL_46;
  }
  v11 = *(_DWORD *)(a1 + 64);
  if ( (v11 & 2) == 0 )
    goto LABEL_30;
LABEL_7:
  if ( (*(_BYTE *)(a2 + 32) & 0xF) == 6 )
  {
LABEL_12:
    v12 = 0;
    goto LABEL_13;
  }
  return 0;
}
