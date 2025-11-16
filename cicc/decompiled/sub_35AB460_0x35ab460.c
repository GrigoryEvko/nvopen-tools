// Function: sub_35AB460
// Address: 0x35ab460
//
__int64 __fastcall sub_35AB460(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r14
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // rbx
  __int64 i; // r13
  unsigned int v13; // eax
  __int64 v15; // rdi
  __int64 v16; // rbx
  int v17; // ecx
  __int64 *v18; // rsi
  signed int v19; // r12d
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // r14
  __int64 v23; // r13
  unsigned int v24; // ebx
  char v25; // al
  __int64 v26; // r15
  unsigned int v27; // ebx
  __int64 v28; // r13
  __int64 v29; // rdi
  int v30; // ecx
  int v31; // eax
  __int64 v32; // rbx
  _BYTE *v33; // r12
  __int64 v34; // rax
  _BYTE *v35; // r15
  _BYTE *v36; // rbx
  char v37; // al
  _BYTE *v38; // r15
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  int v43; // esi
  __int64 v44; // [rsp+0h] [rbp-80h]
  __int64 v45; // [rsp+8h] [rbp-78h]
  __int64 v46; // [rsp+10h] [rbp-70h]
  __int64 v47; // [rsp+18h] [rbp-68h]
  __int64 v48; // [rsp+20h] [rbp-60h]
  __int64 v49; // [rsp+28h] [rbp-58h]
  __int64 v50; // [rsp+30h] [rbp-50h]
  int v51; // [rsp+38h] [rbp-48h]
  __int64 v52[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = a1;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = v3();
  a1[25] = v4;
  v5 = 0;
  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v10 = a2 + 320;
  a1[26] = v6;
  v44 = a2 + 320;
  a1[27] = *(_QWORD *)(a2 + 32);
  v46 = *(_QWORD *)(a2 + 328);
  if ( v46 == a2 + 320 )
    return v5;
  do
  {
    v11 = *(_QWORD *)(v46 + 56);
    for ( i = v46 + 48; i != v11; v11 = *(_QWORD *)(v11 + 8) )
    {
      while ( 1 )
      {
        if ( *(_WORD *)(v11 + 68) == 10 )
        {
          v52[0] = v11;
          sub_35AADF0((__int64)(v2 + 28), v52, v7, v10, v8, v9);
        }
        if ( (*(_BYTE *)v11 & 4) == 0 )
          break;
        v11 = *(_QWORD *)(v11 + 8);
        if ( i == v11 )
          goto LABEL_11;
      }
      while ( (*(_BYTE *)(v11 + 44) & 8) != 0 )
        v11 = *(_QWORD *)(v11 + 8);
    }
LABEL_11:
    v13 = *((_DWORD *)v2 + 66);
    if ( !v13 )
      goto LABEL_12;
    v45 = (__int64)(v2 + 28);
    do
    {
      while ( 1 )
      {
        v15 = v2[29];
        v16 = *(_QWORD *)(v2[32] + 8LL * v13 - 8);
        v7 = *((unsigned int *)v2 + 62);
        v47 = v16;
        if ( (_DWORD)v7 )
        {
          v17 = v7 - 1;
          v7 = ((_DWORD)v7 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v18 = (__int64 *)(v15 + 8 * v7);
          v8 = *v18;
          if ( v16 == *v18 )
          {
LABEL_20:
            *v18 = -8192;
            v13 = *((_DWORD *)v2 + 66);
            --*((_DWORD *)v2 + 60);
            ++*((_DWORD *)v2 + 61);
          }
          else
          {
            v43 = 1;
            while ( v8 != -4096 )
            {
              v9 = (unsigned int)(v43 + 1);
              v7 = v17 & (unsigned int)(v43 + v7);
              v18 = (__int64 *)(v15 + 8LL * (unsigned int)v7);
              v8 = *v18;
              if ( v16 == *v18 )
                goto LABEL_20;
              v43 = v9;
            }
          }
        }
        *((_DWORD *)v2 + 66) = v13 - 1;
        v19 = *(_DWORD *)(*(_QWORD *)(v16 + 32) + 8LL);
        if ( v19 < 0 )
        {
          v28 = *(_QWORD *)(*(_QWORD *)(v2[27] + 56LL) + 16LL * (v19 & 0x7FFFFFFF) + 8);
          if ( !v28 )
            goto LABEL_45;
          while ( (*(_BYTE *)(v28 + 3) & 0x10) != 0 || (*(_BYTE *)(v28 + 4) & 8) != 0 )
          {
            v28 = *(_QWORD *)(v28 + 32);
            if ( !v28 )
              goto LABEL_45;
          }
          while ( 1 )
          {
            v29 = *(_QWORD *)(v28 + 16);
            *(_BYTE *)(v28 + 4) |= 1u;
            v52[0] = v29;
            v30 = *(unsigned __int16 *)(v29 + 68);
            v31 = v30 - 12;
            if ( (((_WORD)v30 - 12) & 0xFFF7) == 0 )
              goto LABEL_53;
            LOBYTE(v31) = (_WORD)v30 == 68;
            if ( (unsigned __int16)v30 <= 0x13u )
              v31 |= (0x80201uLL >> v30) & 1;
            if ( (_BYTE)v31 )
            {
LABEL_53:
              v32 = *(_QWORD *)(v29 + 32);
              v33 = (_BYTE *)(v32 + 40LL * (*(_DWORD *)(v29 + 40) & 0xFFFFFF));
              v34 = 5LL * (unsigned int)sub_2E88FE0(v29);
              if ( v33 != (_BYTE *)(v32 + 8 * v34) )
              {
                v35 = (_BYTE *)(v32 + 8 * v34);
                while ( 1 )
                {
                  v36 = v35;
                  if ( (unsigned __int8)sub_2E2FA70(v35) )
                    break;
                  v35 += 40;
                  if ( v33 == v35 )
                    goto LABEL_66;
                }
                while ( v36 != v33 )
                {
                  v37 = v36[4];
                  if ( (v37 & 1) == 0 && (v37 & 2) == 0 && ((v36[3] & 0x10) == 0 || (*(_DWORD *)v36 & 0xFFF00) != 0) )
                    goto LABEL_67;
                  if ( v36 + 40 == v33 )
                    break;
                  v38 = v36 + 40;
                  while ( 1 )
                  {
                    v36 = v38;
                    if ( (unsigned __int8)sub_2E2FA70(v38) )
                      break;
                    v38 += 40;
                    if ( v33 == v38 )
                      goto LABEL_66;
                  }
                }
              }
LABEL_66:
              sub_2E88D70(v52[0], (unsigned __int16 *)(*(_QWORD *)(v2[25] + 8LL) - 400LL));
              sub_35AADF0(v45, v52, v39, v40, v41, v42);
            }
            do
            {
LABEL_67:
              v28 = *(_QWORD *)(v28 + 32);
              if ( !v28 )
                goto LABEL_45;
            }
            while ( (*(_BYTE *)(v28 + 3) & 0x10) != 0 || (*(_BYTE *)(v28 + 4) & 8) != 0 );
          }
        }
        v20 = *(_QWORD *)(v16 + 8);
        v10 = *(_QWORD *)(v47 + 24) + 48LL;
        v48 = v10;
        if ( v10 != v20 )
          break;
LABEL_40:
        v51 = *(_DWORD *)(v47 + 40);
        v27 = (v51 & 0xFFFFFF) - 1;
        if ( (v51 & 0xFFFFFF) != 1 )
        {
          do
            sub_2E8A650(v47, v27--);
          while ( v27 );
        }
        v13 = *((_DWORD *)v2 + 66);
        if ( !v13 )
          goto LABEL_43;
      }
      v9 = (__int64)v2;
      while ( 1 )
      {
        v21 = *(_QWORD *)(v20 + 32);
        v22 = v21 + 40LL * (*(_DWORD *)(v20 + 40) & 0xFFFFFF);
        if ( v21 != v22 )
          break;
LABEL_38:
        v20 = *(_QWORD *)(v20 + 8);
        if ( v48 == v20 )
        {
          v2 = (_QWORD *)v9;
          goto LABEL_40;
        }
      }
      v49 = v20;
      v23 = v21 + 40;
      v24 = 0;
      while ( 1 )
      {
        v26 = v23 - 40;
        if ( *(_BYTE *)(v23 - 40) )
          break;
        v7 = *(unsigned int *)(v23 - 32);
        if ( (unsigned int)(v7 - 1) > 0x3FFFFFFE )
          break;
        if ( v19 != (_DWORD)v7 )
        {
          if ( (unsigned int)(v19 - 1) > 0x3FFFFFFE )
            break;
          v50 = v9;
          v25 = sub_E92070(*(_QWORD *)(v9 + 208), v19, v7);
          v9 = v50;
          if ( !v25 )
            break;
        }
        if ( (*(_BYTE *)(v26 + 3) & 0x10) == 0 )
          *(_BYTE *)(v26 + 4) |= 1u;
        if ( v22 == v23 )
          goto LABEL_44;
        v24 = 1;
LABEL_33:
        v23 += 40;
      }
      if ( v22 != v23 )
        goto LABEL_33;
      v8 = v24;
      v20 = v49;
      if ( !(_BYTE)v8 )
        goto LABEL_38;
LABEL_44:
      v2 = (_QWORD *)v9;
LABEL_45:
      sub_2E88E20(v47);
      v13 = *((_DWORD *)v2 + 66);
    }
    while ( v13 );
LABEL_43:
    v5 = 1;
LABEL_12:
    v46 = *(_QWORD *)(v46 + 8);
  }
  while ( v44 != v46 );
  return v5;
}
