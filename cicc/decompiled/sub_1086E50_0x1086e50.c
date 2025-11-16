// Function: sub_1086E50
// Address: 0x1086e50
//
unsigned __int64 __fastcall sub_1086E50(__int64 a1, __int64 *a2, __int64 a3)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rcx
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // rcx
  unsigned __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r13
  __int64 v15; // r12
  bool v16; // zf
  int v17; // eax
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rax
  __int64 v30; // rax
  unsigned int v31; // esi
  unsigned int v32; // edi
  __int64 *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rax
  void *v36; // rax
  __int64 v37; // rdi
  void *v38; // rax
  __int64 *v39; // r11
  int v40; // edi
  int v41; // ecx
  int v42; // edi
  int v43; // edi
  unsigned int v44; // eax
  __int64 v45; // r10
  int v46; // esi
  __int64 *v47; // rdx
  int v48; // r8d
  unsigned int v49; // eax
  __int64 *v50; // rdi
  int v51; // edx
  __int64 v52; // rsi
  __int64 v53; // [rsp+0h] [rbp-C0h]
  __int64 v54; // [rsp+8h] [rbp-B8h]
  _QWORD *v55; // [rsp+8h] [rbp-B8h]
  __int64 v56; // [rsp+8h] [rbp-B8h]
  __int64 v57; // [rsp+8h] [rbp-B8h]
  __int64 v58; // [rsp+8h] [rbp-B8h]
  int v59; // [rsp+8h] [rbp-B8h]
  unsigned int v60; // [rsp+8h] [rbp-B8h]
  __int64 v61[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v63[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v64; // [rsp+50h] [rbp-70h]
  void *v65[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v66; // [rsp+80h] [rbp-40h]

  v6 = (_QWORD *)sub_E5C930(a2, a3);
  v7 = v6;
  if ( !v6 )
  {
    v19 = sub_1085B40(a1, a3);
    v13 = 0;
    v15 = v19;
    if ( (*(_BYTE *)(a3 + 13) & 0xE) == 0 )
    {
      *(_DWORD *)(v19 + 12) = -1;
      v14 = v19;
      goto LABEL_8;
    }
LABEL_24:
    *(_BYTE *)(v15 + 18) = 105;
    *(_QWORD *)(v15 + 112) = 0;
    if ( (*(_BYTE *)(a3 + 9) & 0x70) == 0x20 )
    {
      v20 = *(_QWORD *)(a3 + 24);
      *(_BYTE *)(a3 + 8) |= 8u;
      if ( *(_BYTE *)v20 == 2 )
      {
        if ( (v21 = *(_QWORD *)(v20 + 16), !*(_QWORD *)v21)
          && ((*(_BYTE *)(v21 + 9) & 0x70) != 0x20
           || *(char *)(v21 + 8) < 0
           || (*(_BYTE *)(v21 + 8) |= 8u,
               v58 = v13,
               v36 = sub_E807D0(*(_QWORD *)(v21 + 24)),
               v13 = v58,
               (*(_QWORD *)v21 = v36) == 0))
          || (*(_BYTE *)(v21 + 8) & 0x20) != 0 )
        {
          v56 = v13;
          v22 = sub_1085B40(a1, v21);
          v13 = v56;
          if ( v22 )
          {
            v14 = 0;
            goto LABEL_30;
          }
        }
      }
    }
    if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
    {
      v27 = *(__int64 **)(a3 - 8);
      v28 = *v27;
      v29 = v27 + 3;
    }
    else
    {
      v28 = 0;
      v29 = 0;
    }
    v63[2] = v29;
    v63[0] = ".weak.";
    v57 = v13;
    v65[0] = v63;
    v66 = 770;
    v64 = 1283;
    v63[3] = v28;
    v65[2] = ".default";
    sub_CA0F50(v61, v65);
    v30 = sub_1084C60((_QWORD *)a1, v61[0], v61[1]);
    v14 = v30;
    if ( v57 )
      *(_QWORD *)(v30 + 112) = v57;
    else
      *(_DWORD *)(v30 + 12) = -1;
    v31 = *(_DWORD *)(a1 + 232);
    if ( v31 )
    {
      v24 = v31 - 1;
      v23 = *(_QWORD *)(a1 + 216);
      v32 = v24 & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
      v33 = (__int64 *)(v23 + 8LL * v32);
      v34 = *v33;
      if ( v30 == *v33 )
      {
LABEL_45:
        if ( (__int64 *)v61[0] != &v62 )
          j_j___libc_free_0(v61[0], v62 + 1);
        v22 = v14;
LABEL_30:
        *(_QWORD *)(v15 + 104) = v22;
        v25 = *(unsigned int *)(v15 + 72);
        if ( v25 != 1 )
        {
          if ( v25 <= 1 )
          {
            if ( !*(_DWORD *)(v15 + 76) )
            {
              sub_C8D5F0(v15 + 64, (const void *)(v15 + 80), 1u, 0x18u, v23, v24);
              v25 = *(unsigned int *)(v15 + 72);
            }
            v26 = *(_QWORD *)(v15 + 64);
            v35 = v26 + 24 * v25;
            if ( v35 != v26 + 24 )
            {
              do
              {
                if ( v35 )
                {
                  *(_QWORD *)(v35 + 16) = 0;
                  *(_OWORD *)v35 = 0;
                }
                v35 += 24;
              }
              while ( v26 + 24 != v35 );
              v26 = *(_QWORD *)(v15 + 64);
            }
            *(_DWORD *)(v15 + 72) = 1;
LABEL_34:
            *(_QWORD *)(v26 + 16) = 0;
            *(_OWORD *)v26 = 0;
            **(_DWORD **)(v15 + 64) = 0;
            *(_DWORD *)(*(_QWORD *)(v15 + 64) + 4LL) = 0;
            result = (*(unsigned __int16 *)(a3 + 12) >> 9) & 7;
            *(_DWORD *)(*(_QWORD *)(v15 + 64) + 8LL) = result;
            if ( !v14 )
            {
LABEL_16:
              *(_QWORD *)(v15 + 128) = a3;
              return result;
            }
LABEL_8:
            if ( (((*(_BYTE *)(a3 + 9) & 0x70) - 48) & 0xE0) == 0 && (*(_BYTE *)(a3 + 8) & 0x20) != 0 )
            {
              v17 = *(_DWORD *)(a3 + 24);
            }
            else
            {
              v16 = (unsigned __int8)sub_E5BD10((__int64)a2, a3, (__int64)v65) == 0;
              v17 = 0;
              if ( !v16 )
                v17 = (int)v65[0];
            }
            *(_DWORD *)(v14 + 8) = v17;
            *(_WORD *)(v14 + 16) = *(_WORD *)(a3 + 32);
            result = *(unsigned __int16 *)(a3 + 12);
            *(_BYTE *)(v14 + 18) = result;
            if ( !(_BYTE)result )
            {
              v18 = *(_BYTE *)(a3 + 8);
              if ( (v18 & 0x20) == 0
                && (*(_QWORD *)a3
                 || (*(_BYTE *)(a3 + 9) & 0x70) == 0x20
                 && (v18 < 0
                  || (v37 = *(_QWORD *)(a3 + 24),
                      *(_BYTE *)(a3 + 8) = v18 | 8,
                      v38 = sub_E807D0(v37),
                      (*(_QWORD *)a3 = v38) != 0)
                  || (*(_BYTE *)(a3 + 9) & 0x70) == 0x20)) )
              {
                result = 3;
              }
              else
              {
                result = 2;
              }
              *(_BYTE *)(v14 + 18) = result;
            }
            goto LABEL_16;
          }
          *(_DWORD *)(v15 + 72) = 1;
        }
        v26 = *(_QWORD *)(v15 + 64);
        goto LABEL_34;
      }
      v59 = 1;
      v39 = 0;
      v53 = v23;
      while ( v34 != -4096 )
      {
        if ( v34 == -8192 && !v39 )
          v39 = v33;
        v23 = (unsigned int)(v59 + 1);
        v32 = v24 & (v59 + v32);
        v33 = (__int64 *)(v53 + 8LL * v32);
        v34 = *v33;
        if ( v30 == *v33 )
          goto LABEL_45;
        ++v59;
      }
      v40 = *(_DWORD *)(a1 + 224);
      if ( !v39 )
        v39 = v33;
      ++*(_QWORD *)(a1 + 208);
      v41 = v40 + 1;
      if ( 4 * (v40 + 1) < 3 * v31 )
      {
        if ( v31 - *(_DWORD *)(a1 + 228) - v41 > v31 >> 3 )
        {
LABEL_75:
          *(_DWORD *)(a1 + 224) = v41;
          if ( *v39 != -4096 )
            --*(_DWORD *)(a1 + 228);
          *v39 = v14;
          goto LABEL_45;
        }
        v60 = ((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9);
        sub_10852E0(a1 + 208, v31);
        v48 = *(_DWORD *)(a1 + 232);
        if ( v48 )
        {
          v23 = (unsigned int)(v48 - 1);
          v24 = *(_QWORD *)(a1 + 216);
          v49 = v23 & v60;
          v41 = *(_DWORD *)(a1 + 224) + 1;
          v50 = 0;
          v39 = (__int64 *)(v24 + 8LL * ((unsigned int)v23 & v60));
          v51 = 1;
          v52 = *v39;
          if ( v14 != *v39 )
          {
            while ( v52 != -4096 )
            {
              if ( !v50 && v52 == -8192 )
                v50 = v39;
              v49 = v23 & (v49 + v51);
              v39 = (__int64 *)(v24 + 8LL * v49);
              v52 = *v39;
              if ( v14 == *v39 )
                goto LABEL_75;
              ++v51;
            }
            if ( v50 )
              v39 = v50;
          }
          goto LABEL_75;
        }
LABEL_107:
        ++*(_DWORD *)(a1 + 224);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 208);
    }
    sub_10852E0(a1 + 208, 2 * v31);
    v42 = *(_DWORD *)(a1 + 232);
    if ( v42 )
    {
      v43 = v42 - 1;
      v23 = *(_QWORD *)(a1 + 216);
      v44 = v43 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v41 = *(_DWORD *)(a1 + 224) + 1;
      v39 = (__int64 *)(v23 + 8LL * v44);
      v45 = *v39;
      if ( v14 != *v39 )
      {
        v46 = 1;
        v47 = 0;
        while ( v45 != -4096 )
        {
          if ( v45 == -8192 && !v47 )
            v47 = v39;
          v24 = (unsigned int)(v46 + 1);
          v44 = v43 & (v44 + v46);
          v39 = (__int64 *)(v23 + 8LL * v44);
          v45 = *v39;
          if ( v14 == *v39 )
            goto LABEL_75;
          ++v46;
        }
        if ( v47 )
          v39 = v47;
      }
      goto LABEL_75;
    }
    goto LABEL_107;
  }
  v8 = (_QWORD *)*v6;
  if ( !v8 )
  {
    if ( (*((_BYTE *)v7 + 9) & 0x70) != 0x20
      || *((char *)v7 + 8) < 0
      || (*((_BYTE *)v7 + 8) |= 8u, v55 = v7, v8 = sub_E807D0(v7[3]), (*v55 = v8) == 0) )
    {
      v10 = 0;
LABEL_6:
      v54 = v10;
      v12 = sub_1085B40(a1, a3);
      v13 = v54;
      v14 = v12;
      if ( (*(_BYTE *)(a3 + 13) & 0xE) == 0 )
      {
        *(_QWORD *)(v12 + 112) = v54;
        v15 = v12;
        goto LABEL_8;
      }
      v15 = v12;
      goto LABEL_24;
    }
  }
  v9 = (_QWORD *)v8[1];
  v65[0] = v9;
  v10 = *sub_1085900(a1 + 144, (__int64 *)v65);
  if ( *(_DWORD *)(a1 + 244) != 1 )
    goto LABEL_6;
  if ( !v9 )
    goto LABEL_6;
  result = v9[17];
  if ( result <= 3 || *(_DWORD *)(v9[16] + result - 4) != 1870095406 )
    goto LABEL_6;
  return result;
}
