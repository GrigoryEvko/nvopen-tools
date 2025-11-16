// Function: sub_2598870
// Address: 0x2598870
//
bool __fastcall sub_2598870(__int64 *a1, unsigned __int8 *a2)
{
  __int64 *v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  int v12; // r11d
  int v13; // r14d
  unsigned int v14; // ebx
  int v15; // r13d
  char v16; // al
  char v17; // r11
  char v18; // al
  int v20; // eax
  int v21; // eax
  int v22; // edx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r14
  int v27; // r14d
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v35; // r13
  __int64 v36; // rsi
  int v37; // ecx
  unsigned __int8 *v38; // rdx
  __int64 v39; // rax
  int v40; // eax
  int v41; // [rsp+Ch] [rbp-94h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  __int64 v43; // [rsp+28h] [rbp-78h]
  int v44; // [rsp+30h] [rbp-70h]
  unsigned __int64 v45; // [rsp+30h] [rbp-70h]
  __int64 v46; // [rsp+38h] [rbp-68h]
  void *v47; // [rsp+40h] [rbp-60h] BYREF
  __int64 v48; // [rsp+48h] [rbp-58h]
  void **v49; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int8 *v50; // [rsp+58h] [rbp-48h]
  __int64 v51; // [rsp+60h] [rbp-40h]
  __int64 v52; // [rsp+68h] [rbp-38h]

  v3 = a1;
  v4 = *a1;
  v5 = a1[2];
  v46 = a1[1];
  v43 = *a1;
  v47 = &unk_4A16F98;
  v48 = 0xFF00000000LL;
  if ( (unsigned __int8)(*a2 - 34) <= 0x33u )
  {
    v10 = 0x8000000000041LL;
    if ( _bittest64(&v10, (unsigned int)*a2 - 34) )
    {
      sub_250D230((unsigned __int64 *)&v49, (unsigned __int64)a2, 5, 0);
      v11 = sub_252A070(v4, (__int64)v49, (__int64)v50, v5, 1, 0, 1);
      if ( !v11 )
      {
        v40 = sub_2537A90((__int64)a2);
        sub_2561E50(v5, (__int64)&v47, 0x80u, (__int64)a2, 0, v46, v40);
        v9 = -128;
        goto LABEL_19;
      }
      v12 = *(_DWORD *)(v11 + 100);
      if ( (unsigned __int8)v12 == 255 || (*(_DWORD *)(v11 + 100) & 0xFC) == 0xFC )
      {
        v9 = -1;
        goto LABEL_19;
      }
      if ( (*(_DWORD *)(v11 + 100) & 0xDC) != 0xDC )
      {
        v42 = v11;
        v13 = 8;
        v41 = *(_DWORD *)(v11 + 100);
        v44 = v12 | 0x1C;
        v14 = 1;
        do
        {
          if ( (v14 & v44) == 0 )
          {
            v15 = (unsigned __int8)sub_B46420((__int64)a2);
            v16 = sub_B46490((__int64)a2);
            sub_2561E50(v5, (__int64)&v47, v14, (__int64)a2, 0, v46, (2 * (v16 != 0)) | v15);
          }
          v14 *= 2;
          --v13;
        }
        while ( v13 );
        v17 = v41;
        v3 = a1;
        if ( (~(_BYTE)v41 & 0xC) != 0 )
        {
          v50 = a2;
          v49 = &v47;
          v52 = v5;
          v51 = v46;
          v18 = (*(__int64 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64 *, __int64, __int64, __int64, unsigned int), void ***, __int64))(*(_QWORD *)v42 + 120LL))(
                  v42,
                  sub_25621E0,
                  &v49,
                  243);
          v17 = v41;
          if ( !v18 )
          {
            v9 = -256;
            goto LABEL_19;
          }
        }
        if ( (v17 & 0x10) != 0 )
        {
LABEL_21:
          v9 = BYTE4(v48) | 0xFFFFFF00;
          goto LABEL_19;
        }
        v22 = *a2;
        if ( v22 == 40 )
        {
          v23 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
        }
        else
        {
          v23 = 0;
          if ( v22 != 85 )
          {
            v23 = 64;
            if ( v22 != 34 )
              BUG();
          }
        }
        if ( (a2[7] & 0x80u) != 0 )
        {
          v24 = sub_BD2BC0((__int64)a2);
          v26 = v24 + v25;
          if ( (a2[7] & 0x80u) == 0 )
          {
            if ( (unsigned int)(v26 >> 4) )
              goto LABEL_53;
          }
          else if ( (unsigned int)((v26 - sub_BD2BC0((__int64)a2)) >> 4) )
          {
            if ( (a2[7] & 0x80u) != 0 )
            {
              v27 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
              if ( (a2[7] & 0x80u) == 0 )
                BUG();
              v28 = sub_BD2BC0((__int64)a2);
              v30 = 32LL * (unsigned int)(*(_DWORD *)(v28 + v29 - 4) - v27);
              goto LABEL_33;
            }
LABEL_53:
            BUG();
          }
        }
        v30 = 0;
LABEL_33:
        v31 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
        v32 = (32 * v31 - 32 - v23 - v30) >> 5;
        if ( (_DWORD)v32 )
        {
          v33 = (unsigned int)v32;
          v35 = 0;
          while ( 1 )
          {
            v36 = *(_QWORD *)(*(_QWORD *)&a2[32 * (v35 - v31)] + 8LL);
            v37 = *(unsigned __int8 *)(v36 + 8);
            if ( (unsigned int)(v37 - 17) <= 1 )
              LOBYTE(v37) = *(_BYTE *)(**(_QWORD **)(v36 + 16) + 8LL);
            if ( (_BYTE)v37 == 14 )
            {
              if ( (a2[7] & 0x40) != 0 )
                v38 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
              else
                v38 = &a2[-32 * v31];
              v45 = *(_QWORD *)&a2[32 * (v35 - v31)];
              v50 = 0;
              v49 = (void **)((unsigned __int64)&v38[32 * v35] | 3);
              nullsub_1518();
              v39 = sub_25294B0(v43, (__int64)v49, (__int64)v50, v5, 1, 0, 1);
              if ( !v39 || (*(_BYTE *)(v39 + 97) & 3) != 3 )
                sub_2597480(v5, v43, (__int64)a2, v45, (__int64)&v47, v46, 0);
            }
            if ( ++v35 == v33 )
              break;
            v31 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
          }
          v3 = a1;
        }
        goto LABEL_21;
      }
      v21 = sub_2537A90((__int64)a2);
      sub_2561E50(v5, (__int64)&v47, 0x20u, (__int64)a2, 0, v46, v21);
LABEL_23:
      v9 = BYTE4(v48) | 0xFFFFFF00;
      goto LABEL_19;
    }
  }
  v6 = sub_2537AD0(a2);
  v7 = v6;
  if ( !v6 )
  {
    v20 = sub_2537A90((__int64)a2);
    sub_2561E50(v5, (__int64)&v47, 0x80u, (__int64)a2, 0, v46, v20);
    goto LABEL_23;
  }
  v8 = *(_QWORD *)(v6 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  sub_2597480(v5, v43, (__int64)a2, v7, (__int64)&v47, v46, BYTE1(*(_DWORD *)(v8 + 8)));
  v9 = BYTE4(v48) | 0xFFFFFF00;
LABEL_19:
  *(_DWORD *)(v3[2] + 100) = *(_DWORD *)(v3[2] + 96) | *(_DWORD *)(v3[2] + 100) & v9;
  return *(_DWORD *)(v3[2] + 100) != 256;
}
