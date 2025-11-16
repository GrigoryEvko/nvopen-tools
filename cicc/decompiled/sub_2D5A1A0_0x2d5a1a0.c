// Function: sub_2D5A1A0
// Address: 0x2d5a1a0
//
__int64 __fastcall sub_2D5A1A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r9d
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  char *v8; // r12
  __int64 v9; // rbx
  __int64 *v10; // rsi
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rsi
  unsigned int v14; // edx
  _QWORD *v15; // rbx
  unsigned int v16; // ecx
  __int64 v17; // rsi
  _QWORD *v18; // r12
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  bool v24; // zf
  __int64 v26; // rax
  _QWORD *v27; // rbx
  _QWORD *v28; // r12
  __int64 v29; // rsi
  int v30; // r14d
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  int v34; // eax
  unsigned int v35; // edx
  unsigned int v36; // eax
  _QWORD *v37; // r12
  unsigned __int64 v38; // rdi
  __int64 v39; // rdi
  _QWORD *v40; // rax
  __int64 v41; // rdx
  _QWORD *v42; // rdx
  char v43; // cl
  _QWORD *v44; // rbx
  char v45; // al
  __int64 v46; // rax
  __int64 v47; // [rsp+18h] [rbp-C8h]
  int v48; // [rsp+20h] [rbp-C0h]
  int v49; // [rsp+20h] [rbp-C0h]
  __int64 v50; // [rsp+28h] [rbp-B8h]
  __int64 v51; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+38h] [rbp-A8h]
  __int64 v53; // [rsp+40h] [rbp-A0h]
  void *v54; // [rsp+50h] [rbp-90h]
  _QWORD v55[2]; // [rsp+58h] [rbp-88h] BYREF
  __int64 v56; // [rsp+68h] [rbp-78h]
  __int64 v57; // [rsp+70h] [rbp-70h]
  void *v58; // [rsp+80h] [rbp-60h] BYREF
  __int64 v59; // [rsp+88h] [rbp-58h] BYREF
  void (__fastcall *v60)(void **, void **, __int64); // [rsp+90h] [rbp-50h]
  __int64 v61; // [rsp+98h] [rbp-48h]
  __int64 i; // [rsp+A0h] [rbp-40h]

  v2 = 0;
  v47 = a2 + 72;
  v50 = *(_QWORD *)(a2 + 80);
  if ( v50 != a2 + 72 )
  {
    do
    {
      if ( !v50 )
        BUG();
      v4 = *(_QWORD *)(v50 + 32);
      v5 = v50 + 24;
      *(_WORD *)(a1 + 96) = 1;
      *(_QWORD *)(a1 + 88) = v4;
      if ( v50 + 24 == v4 )
        goto LABEL_52;
      do
      {
        while ( 1 )
        {
          v6 = v4;
          v4 = *(_QWORD *)(v4 + 8);
          *(_WORD *)(a1 + 96) = 0;
          *(_QWORD *)(a1 + 88) = v4;
          if ( *(_BYTE *)(v6 - 24) == 85 )
          {
            v7 = *(_QWORD *)(v6 - 56);
            if ( v7 )
            {
              if ( !*(_BYTE *)v7
                && *(_QWORD *)(v7 + 24) == *(_QWORD *)(v6 + 56)
                && (*(_BYTE *)(v7 + 33) & 0x20) != 0
                && *(_DWORD *)(v7 + 36) == 11 )
              {
                break;
              }
            }
          }
          if ( v4 == v5 )
            goto LABEL_52;
        }
        v8 = *(char **)(v6 - 32LL * (*(_DWORD *)(v6 - 20) & 0x7FFFFFF) - 24);
        sub_B43D60((_QWORD *)(v6 - 24));
        v9 = *(_QWORD *)(a1 + 88);
        if ( v9 )
        {
          v51 = 6;
          v9 -= 24;
          v52 = 0;
          v53 = v9;
          if ( v9 != -4096 && v9 != -8192 )
            sub_BD73F0((__int64)&v51);
        }
        else
        {
          v51 = 6;
          v52 = 0;
          v53 = 0;
        }
        v10 = *(__int64 **)(a1 + 48);
        v60 = 0;
        sub_F5CAB0(v8, v10, 0, (__int64)&v58);
        if ( v60 )
          v60(&v58, &v58, 3);
        if ( v53 == v9 )
          goto LABEL_48;
        v11 = *(_QWORD *)(v50 + 32);
        ++*(_QWORD *)(a1 + 104);
        *(_WORD *)(a1 + 96) = 1;
        *(_QWORD *)(a1 + 88) = v11;
        v12 = *(_DWORD *)(a1 + 120);
        if ( !v12 && !*(_DWORD *)(a1 + 124) )
          goto LABEL_46;
        v13 = *(unsigned int *)(a1 + 128);
        v14 = 4 * v12;
        v15 = *(_QWORD **)(a1 + 112);
        v55[0] = 2;
        v16 = v13;
        v17 = v13 << 6;
        v55[1] = 0;
        if ( (unsigned int)(4 * v12) < 0x40 )
          v14 = 64;
        v56 = -4096;
        v18 = (_QWORD *)((char *)v15 + v17);
        if ( v16 > v14 )
        {
          v30 = v12;
          v57 = 0;
          v59 = 2;
          v54 = &unk_4A26638;
          v58 = &unk_4A26638;
          v31 = -4096;
          v60 = 0;
          v61 = -8192;
          i = 0;
          while ( 1 )
          {
            v32 = v15[3];
            if ( v32 != v31 )
            {
              v31 = v61;
              if ( v32 != v61 )
              {
                v33 = v15[7];
                if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
                {
                  sub_BD60C0(v15 + 5);
                  v32 = v15[3];
                }
                v31 = v32;
              }
            }
            *v15 = &unk_49DB368;
            if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
              sub_BD60C0(v15 + 1);
            v15 += 8;
            if ( v15 == v18 )
              break;
            v31 = v56;
          }
          v34 = v30;
          v58 = &unk_49DB368;
          if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
          {
            sub_BD60C0(&v59);
            v34 = v30;
          }
          v54 = &unk_49DB368;
          if ( v56 != 0 && v56 != -4096 && v56 != -8192 )
          {
            v48 = v34;
            sub_BD60C0(v55);
            v34 = v48;
          }
          if ( v34 )
          {
            v35 = v34 - 1;
            v34 = 64;
            if ( v35 )
            {
              _BitScanReverse(&v36, v35);
              v34 = 1 << (33 - (v36 ^ 0x1F));
              if ( v34 < 64 )
                v34 = 64;
            }
          }
          v37 = *(_QWORD **)(a1 + 112);
          if ( *(_DWORD *)(a1 + 128) == v34 )
          {
            *(_QWORD *)(a1 + 120) = 0;
            v59 = 2;
            v44 = &v37[8 * (unsigned __int64)(unsigned int)v34];
            v60 = 0;
            v61 = -4096;
            v58 = &unk_4A26638;
            i = 0;
            if ( v44 != v37 )
            {
              do
              {
                if ( v37 )
                {
                  v45 = v59;
                  v37[2] = 0;
                  v37[1] = v45 & 6;
                  v46 = v61;
                  v24 = v61 == 0;
                  v37[3] = v61;
                  if ( v46 != -4096 && !v24 && v46 != -8192 )
                    sub_BD6050(v37 + 1, v59 & 0xFFFFFFFFFFFFFFF8LL);
                  *v37 = &unk_4A26638;
                  v37[4] = i;
                }
                v37 += 8;
              }
              while ( v44 != v37 );
              v58 = &unk_49DB368;
              if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
                sub_BD60C0(&v59);
            }
          }
          else
          {
            v49 = v34;
            sub_C7D6A0(*(_QWORD *)(a1 + 112), v17, 8);
            if ( v49 )
            {
              v38 = (((((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
                      | (4 * v49 / 3u + 1)
                      | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
                    | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
                    | (4 * v49 / 3u + 1)
                    | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 8)
                  | (((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
                    | (4 * v49 / 3u + 1)
                    | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
                  | (4 * v49 / 3u + 1)
                  | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1);
              v39 = ((v38 >> 16) | v38) + 1;
              *(_DWORD *)(a1 + 128) = v39;
              v40 = (_QWORD *)sub_C7D670(v39 << 6, 8);
              v41 = *(unsigned int *)(a1 + 128);
              *(_QWORD *)(a1 + 120) = 0;
              *(_QWORD *)(a1 + 112) = v40;
              v59 = 2;
              v42 = &v40[8 * v41];
              v60 = 0;
              v61 = -4096;
              v58 = &unk_4A26638;
              for ( i = 0; v42 != v40; v40 += 8 )
              {
                if ( v40 )
                {
                  v43 = v59;
                  v40[2] = 0;
                  v40[3] = -4096;
                  *v40 = &unk_4A26638;
                  v40[1] = v43 & 6;
                  v40[4] = i;
                }
              }
            }
            else
            {
              *(_QWORD *)(a1 + 112) = 0;
              *(_QWORD *)(a1 + 120) = 0;
              *(_DWORD *)(a1 + 128) = 0;
            }
          }
          goto LABEL_46;
        }
        v57 = 0;
        v19 = -4096;
        v59 = 2;
        v60 = 0;
        v54 = &unk_4A26638;
        v58 = &unk_4A26638;
        v20 = -8192;
        v61 = -8192;
        i = 0;
        if ( v15 == v18 )
          goto LABEL_43;
        while ( 1 )
        {
          v22 = v15[3];
          if ( v22 != v19 )
          {
            if ( v22 == v20 )
              goto LABEL_34;
            v21 = v15[7];
            if ( v21 == 0 || v21 == -4096 || v21 == -8192 )
            {
              v20 = v15[3];
              goto LABEL_34;
            }
            sub_BD60C0(v15 + 5);
            v20 = v15[3];
            if ( v20 != v56 )
            {
LABEL_34:
              if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
                sub_BD60C0(v15 + 1);
              v23 = v56;
              v24 = v56 == 0;
              v15[3] = v56;
              if ( v23 != -4096 && !v24 && v23 != -8192 )
                sub_BD6050(v15 + 1, v55[0] & 0xFFFFFFFFFFFFFFF8LL);
            }
            v15[4] = v57;
            v20 = v61;
          }
          v15 += 8;
          if ( v15 == v18 )
            break;
          v19 = v56;
        }
        v58 = &unk_49DB368;
        if ( v20 != -8192 && v20 != -4096 && v20 )
          sub_BD60C0(&v59);
LABEL_43:
        *(_QWORD *)(a1 + 120) = 0;
        v54 = &unk_49DB368;
        if ( v56 != -4096 && v56 != 0 && v56 != -8192 )
          sub_BD60C0(v55);
LABEL_46:
        if ( *(_BYTE *)(a1 + 168) )
        {
          v26 = *(unsigned int *)(a1 + 160);
          *(_BYTE *)(a1 + 168) = 0;
          if ( (_DWORD)v26 )
          {
            v27 = *(_QWORD **)(a1 + 144);
            v28 = &v27[2 * v26];
            do
            {
              if ( *v27 != -8192 && *v27 != -4096 )
              {
                v29 = v27[1];
                if ( v29 )
                  sub_B91220((__int64)(v27 + 1), v29);
              }
              v27 += 2;
            }
            while ( v28 != v27 );
            LODWORD(v26) = *(_DWORD *)(a1 + 160);
          }
          sub_C7D6A0(*(_QWORD *)(a1 + 144), 16LL * (unsigned int)v26, 8);
        }
        v9 = v53;
LABEL_48:
        if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
          sub_BD60C0(&v51);
        v4 = *(_QWORD *)(a1 + 88);
        v2 = 1;
      }
      while ( v5 != v4 );
LABEL_52:
      v50 = *(_QWORD *)(v50 + 8);
    }
    while ( v47 != v50 );
  }
  return v2;
}
