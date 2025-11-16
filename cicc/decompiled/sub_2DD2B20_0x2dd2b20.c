// Function: sub_2DD2B20
// Address: 0x2dd2b20
//
__int64 __fastcall sub_2DD2B20(_QWORD *a1, __int64 *a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 (*v9)(void); // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 (*v18)(void); // rax
  __int64 v19; // r15
  char *v20; // r12
  char *v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rbx
  __int64 v24; // r15
  char *v25; // r13
  char *v26; // r14
  __int64 v27; // rcx
  __int64 v28; // rbx
  int v29; // eax
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 *v34; // r14
  __int64 v35; // rax
  unsigned __int8 *v36; // rsi
  __int64 v37; // r15
  __int64 v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int64 *v41; // r13
  unsigned __int8 **v42; // r14
  __int64 v43; // [rsp+8h] [rbp-D8h]
  __int64 v44; // [rsp+18h] [rbp-C8h]
  __int64 v45; // [rsp+20h] [rbp-C0h]
  __int64 v46; // [rsp+28h] [rbp-B8h]
  __int64 v47; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v48; // [rsp+38h] [rbp-A8h]
  __int64 v49; // [rsp+48h] [rbp-98h]
  __int64 v50; // [rsp+48h] [rbp-98h]
  unsigned __int8 *v51; // [rsp+58h] [rbp-88h] BYREF
  unsigned __int8 *v52; // [rsp+60h] [rbp-80h] BYREF
  __int64 v53; // [rsp+68h] [rbp-78h]
  __int64 v54; // [rsp+70h] [rbp-70h]
  __int64 v55; // [rsp+80h] [rbp-60h] BYREF
  int v56; // [rsp+88h] [rbp-58h]
  __int64 v57; // [rsp+90h] [rbp-50h]
  unsigned __int8 *v58; // [rsp+98h] [rbp-48h]
  int v59; // [rsp+A0h] [rbp-40h]

  if ( (*(_BYTE *)(*a2 + 3) & 0x40) == 0 )
    return 0;
  v3 = (__int64 *)a1[1];
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_74:
    BUG();
  v6 = (__int64)a2;
  while ( *(_UNKNOWN **)v4 != &unk_501DA08 )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_74;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_501DA08);
  v8 = *a2;
  a1[25] = sub_2DD15F0(v7, *(_QWORD *)v6);
  v9 = *(__int64 (**)(void))(**(_QWORD **)(v6 + 16) + 128LL);
  v10 = 0;
  if ( v9 != sub_2DAC790 )
    v10 = v9();
  a1[26] = v10;
  v11 = *(_QWORD *)(v6 + 48);
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 16) + 200LL))(*(_QWORD *)(v6 + 16));
  v14 = v12;
  if ( *(_BYTE *)(v11 + 36)
    || (v8 = v6, (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v12 + 544LL))(v12, v6))
    && (v8 = v6, (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 536LL))(v14, v6)) )
  {
    v15 = -1;
    v16 = a1[25];
  }
  else
  {
    v15 = *(_QWORD *)(v11 + 48);
    v16 = a1[25];
  }
  *(_QWORD *)(v16 + 16) = v15;
  if ( *(_BYTE *)(*(_QWORD *)(a1[25] + 8LL) + 42LL) )
  {
    v27 = v6 + 320;
    v44 = v6 + 320;
    v46 = *(_QWORD *)(v6 + 328);
    if ( v46 != v6 + 320 )
    {
      v43 = v6;
      while ( 1 )
      {
        v50 = v46 + 48;
        if ( *(_QWORD *)(v46 + 56) != v46 + 48 )
          break;
LABEL_62:
        v46 = *(_QWORD *)(v46 + 8);
        if ( v44 == v46 )
        {
          v6 = v43;
          goto LABEL_12;
        }
      }
      v28 = *(_QWORD *)(v46 + 56);
      while ( 1 )
      {
        v29 = *(_DWORD *)(v28 + 44);
        if ( (v29 & 4) != 0 || (v29 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL) & 0x80u) != 0LL )
          {
LABEL_32:
            v30 = *(_DWORD *)(v28 + 44);
            if ( (v30 & 4) != 0 || (v30 & 8) == 0 )
            {
              v31 = (*(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL) >> 9) & 1LL;
            }
            else
            {
              v8 = 512;
              LOBYTE(v31) = sub_2E88A90(v28, 512, 1);
            }
            if ( (_BYTE)v31 )
              goto LABEL_28;
            v32 = v28;
            if ( (*(_BYTE *)v28 & 4) == 0 && (*(_BYTE *)(v28 + 44) & 8) != 0 )
            {
              do
                v32 = *(_QWORD *)(v32 + 8);
              while ( (*(_BYTE *)(v32 + 44) & 8) != 0 );
            }
            v33 = *(_QWORD *)(v28 + 24);
            v34 = *(__int64 **)(v32 + 8);
            v47 = v33;
            v35 = sub_E6C430(*(_QWORD *)(*(_QWORD *)(v33 + 32) + 24LL), v8, v15, v27, v13);
            v36 = *(unsigned __int8 **)(v28 + 56);
            v48 = (unsigned __int8 *)v35;
            v51 = v36;
            v45 = *(_QWORD *)(a1[26] + 8LL) - 200LL;
            if ( v36 )
            {
              sub_B96E90((__int64)&v51, (__int64)v36, 1);
              v52 = v51;
              if ( v51 )
              {
                sub_B976B0((__int64)&v51, v51, (__int64)&v52);
                v53 = 0;
                v54 = 0;
                v37 = *(_QWORD *)(v33 + 32);
                v51 = 0;
                v55 = (__int64)v52;
                if ( v52 )
                  sub_B96E90((__int64)&v55, (__int64)v52, 1);
LABEL_43:
                v38 = sub_2E7B380(v37, v45, &v55, 0);
                if ( v55 )
                  sub_B91220((__int64)&v55, v55);
                sub_2E31040(v47 + 40, v38);
                v39 = *v34;
                v40 = *(_QWORD *)v38;
                *(_QWORD *)(v38 + 8) = v34;
                v39 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)v38 = v39 | v40 & 7;
                *(_QWORD *)(v39 + 8) = v38;
                *v34 = v38 | *v34 & 7;
                if ( v53 )
                  sub_2E882B0(v38, v37);
                if ( v54 )
                  sub_2E88680(v38, v37);
                LOBYTE(v55) = 15;
                v57 = 0;
                LODWORD(v55) = v55 & 0xFFF000FF;
                v58 = v48;
                v56 = 0;
                v59 = 0;
                sub_2E8EAD0(v38, v37, &v55);
                if ( v52 )
                  sub_B91220((__int64)&v52, (__int64)v52);
                if ( v51 )
                  sub_B91220((__int64)&v51, (__int64)v51);
                v41 = (unsigned __int64 *)a1[25];
                v52 = v48;
                v42 = (unsigned __int8 **)v41[7];
                if ( v42 == (unsigned __int8 **)v41[8] )
                {
                  v8 = v41[7];
                  sub_2DD28A0(v41 + 6, (_QWORD *)v8, &v52, (__int64 *)(v28 + 56));
                  goto LABEL_28;
                }
                v8 = *(_QWORD *)(v28 + 56);
                v55 = v8;
                if ( v8 )
                {
                  sub_B96E90((__int64)&v55, v8, 1);
                  if ( !v42 )
                  {
                    v8 = v55;
                    if ( v55 )
                      sub_B91220((__int64)&v55, v55);
                    goto LABEL_58;
                  }
                }
                else if ( !v42 )
                {
LABEL_58:
                  v41[7] += 16LL;
                  goto LABEL_28;
                }
                *v42 = v48;
                v8 = v55;
                v42[1] = (unsigned __int8 *)v55;
                if ( v8 )
                  sub_B976B0((__int64)&v55, (unsigned __int8 *)v8, (__int64)(v42 + 1));
                goto LABEL_58;
              }
            }
            else
            {
              v52 = 0;
            }
            v53 = 0;
            v54 = 0;
            v37 = *(_QWORD *)(v33 + 32);
            v55 = 0;
            goto LABEL_43;
          }
        }
        else
        {
          v8 = 128;
          if ( (unsigned __int8)sub_2E88A90(v28, 128, 1) )
            goto LABEL_32;
        }
LABEL_28:
        if ( (*(_BYTE *)v28 & 4) != 0 )
        {
          v28 = *(_QWORD *)(v28 + 8);
          if ( v50 == v28 )
            goto LABEL_62;
        }
        else
        {
          while ( (*(_BYTE *)(v28 + 44) & 8) != 0 )
            v28 = *(_QWORD *)(v28 + 8);
          v28 = *(_QWORD *)(v28 + 8);
          if ( v50 == v28 )
            goto LABEL_62;
        }
      }
    }
  }
LABEL_12:
  v17 = 0;
  v18 = *(__int64 (**)(void))(**(_QWORD **)(v6 + 16) + 136LL);
  if ( v18 != sub_2DD19D0 )
    v17 = v18();
  v19 = a1[25];
  v20 = *(char **)(v19 + 24);
  v21 = *(char **)(v19 + 32);
  if ( v20 != v21 )
  {
    v22 = a1[25];
    v23 = v17;
    v24 = v6;
    v25 = v20;
    do
    {
      v26 = v25 + 16;
      if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v24 + 48) + 8LL)
                     + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(v24 + 48) + 32LL) + *(_DWORD *)v25)
                     + 8) == -1 )
      {
        if ( v26 != v21 )
        {
          v49 = v22;
          memmove(v25, v25 + 16, v21 - v26);
          v22 = v49;
        }
        *(_QWORD *)(v22 + 32) -= 16LL;
      }
      else
      {
        LODWORD(v55) = 0;
        *((_DWORD *)v25 + 1) = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64 *))(*(_QWORD *)v23 + 224LL))(
                                 v23,
                                 v24,
                                 *(unsigned int *)v25,
                                 &v55);
        v25 += 16;
      }
      v22 = a1[25];
      v21 = *(char **)(v22 + 32);
    }
    while ( v21 != v25 );
  }
  return 0;
}
