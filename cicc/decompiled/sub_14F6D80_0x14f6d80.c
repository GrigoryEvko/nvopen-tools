// Function: sub_14F6D80
// Address: 0x14f6d80
//
__int64 *__fastcall sub_14F6D80(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  const char *v3; // rax
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  _BOOL8 v17; // rdi
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // rdx
  int v21; // r13d
  unsigned __int64 v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // rbx
  int i; // r15d
  const char *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rax
  char *v32; // rsi
  _QWORD *v33; // [rsp+0h] [rbp-330h]
  __int64 v34; // [rsp+8h] [rbp-328h]
  __int64 v35; // [rsp+10h] [rbp-320h]
  __int64 v36; // [rsp+10h] [rbp-320h]
  __int64 v37; // [rsp+10h] [rbp-320h]
  __int64 v38; // [rsp+28h] [rbp-308h]
  int v39; // [rsp+28h] [rbp-308h]
  __int64 v40; // [rsp+30h] [rbp-300h]
  _BYTE *v42; // [rsp+40h] [rbp-2F0h] BYREF
  __int64 v43; // [rsp+48h] [rbp-2E8h]
  _BYTE v44[64]; // [rsp+50h] [rbp-2E0h] BYREF
  _QWORD v45[2]; // [rsp+90h] [rbp-2A0h] BYREF
  int v46; // [rsp+A0h] [rbp-290h] BYREF
  _QWORD *v47; // [rsp+A8h] [rbp-288h]
  int *v48; // [rsp+B0h] [rbp-280h]
  int *v49; // [rsp+B8h] [rbp-278h]
  __int64 v50; // [rsp+C0h] [rbp-270h]
  __int64 v51; // [rsp+C8h] [rbp-268h]
  __int64 v52; // [rsp+D0h] [rbp-260h]
  __int64 v53; // [rsp+D8h] [rbp-258h]
  __int64 v54; // [rsp+E0h] [rbp-250h]
  __int64 v55; // [rsp+E8h] [rbp-248h]
  char *v56; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v57; // [rsp+F8h] [rbp-238h]
  char v58; // [rsp+100h] [rbp-230h] BYREF
  char v59; // [rsp+101h] [rbp-22Fh]

  v2 = a2;
  v40 = a2 + 32;
  if ( (unsigned __int8)sub_15127D0(a2 + 32, 9, 0) )
  {
    v59 = 1;
    v3 = "Invalid record";
    goto LABEL_4;
  }
  if ( *(_QWORD *)(a2 + 1304) != *(_QWORD *)(a2 + 1296) )
  {
    v59 = 1;
    v3 = "Invalid multiple blocks";
LABEL_4:
    v56 = (char *)v3;
    v58 = 3;
    sub_14EE4B0(a1, a2 + 8, (__int64)&v56);
    return a1;
  }
  v56 = &v58;
  v57 = 0x4000000000LL;
  v42 = v44;
  v43 = 0x800000000LL;
  while ( 1 )
  {
    v5 = sub_14ED070(v40, 0);
    if ( (_DWORD)v5 == 1 )
      break;
    if ( (v5 & 0xFFFFFFFD) == 0 )
    {
      BYTE1(v46) = 1;
      v26 = "Malformed block";
LABEL_41:
      v45[0] = v26;
      LOBYTE(v46) = 3;
      sub_14EE4B0(a1, v2 + 8, (__int64)v45);
      goto LABEL_42;
    }
    LODWORD(v57) = 0;
    v6 = sub_1510D70(v40, HIDWORD(v5), &v56, 0);
    if ( v6 == 1 )
    {
      v39 = v57;
      if ( (v57 & 1) != 0 )
      {
        BYTE1(v46) = 1;
        v26 = "Invalid record";
        goto LABEL_41;
      }
      if ( (_DWORD)v57 )
      {
        v36 = v2;
        v21 = 0;
        do
        {
          v45[0] = 0;
          v46 = 0;
          v48 = &v46;
          v49 = &v46;
          v47 = 0;
          v50 = 0;
          v51 = 0;
          v52 = 0;
          v53 = 0;
          v54 = 0;
          v55 = 0;
          v22 = *(_QWORD *)&v56[8 * (v21 + 1)];
          v23 = WORD1(v22);
          if ( (_DWORD)v23 )
            sub_1560C00(v45, v23);
          v24 = (v22 >> 11) & 0x1FFFFE00000LL | (unsigned __int16)v22;
          if ( v24 )
          {
            for ( i = 1; ; ++i )
            {
              if ( ((i - 3) & 0xFFFFFFFD) != 0 && (unsigned int)(i - 10) > 1 )
              {
                switch ( i )
                {
                  case 1:
                  case 6:
                  case 9:
                  case 12:
                  case 14:
                  case 15:
                  case 17:
                  case 24:
                  case 28:
                  case 33:
                  case 34:
                  case 36:
                  case 42:
                  case 44:
                  case 47:
                  case 48:
                  case 53:
                  case 55:
                  case 56:
                  case 58:
                    break;
                  case 2:
                    if ( (v24 & 0x1F0000) != 0 )
                      sub_1560C00(v45, 0x8000000000000000LL);
                    continue;
                  case 3:
                  case 5:
                  case 10:
                  case 11:
                  case 59:
                    if ( (v24 & 1) == 0 )
                      goto LABEL_88;
                    goto LABEL_49;
                  case 4:
                    v28 = 4096;
                    goto LABEL_53;
                  case 7:
                    v28 = 128;
                    goto LABEL_53;
                  case 8:
                    v28 = 0x10000000000LL;
                    goto LABEL_53;
                  case 13:
                    if ( (v24 & 8) == 0 )
                      continue;
                    goto LABEL_49;
                  case 16:
                    v28 = 0x2000000;
                    goto LABEL_53;
                  case 18:
                    v28 = 0x200000000LL;
                    goto LABEL_53;
                  case 19:
                    v28 = (__int64)&loc_1000000;
                    goto LABEL_53;
                  case 20:
                    v28 = 256;
                    goto LABEL_53;
                  case 21:
                    v27 = 64;
                    goto LABEL_48;
                  case 22:
                    v27 = 0x4000000000LL;
                    goto LABEL_48;
                  case 23:
                    v27 = 0x200000;
                    goto LABEL_48;
                  case 25:
                    v27 = 0x400000000LL;
                    goto LABEL_48;
                  case 26:
                    v27 = 0x800000;
                    goto LABEL_48;
                  case 27:
                    v27 = 2048;
                    goto LABEL_48;
                  case 29:
                    v27 = (__int64)&dword_400000;
                    goto LABEL_48;
                  case 30:
                    v27 = 4;
                    goto LABEL_48;
                  case 31:
                    if ( (v24 & 0x20) == 0 )
                      break;
                    goto LABEL_49;
                  case 32:
                    v27 = 0x80000000LL;
                    goto LABEL_48;
                  case 35:
                    v27 = 0x2000;
                    goto LABEL_48;
                  case 37:
                    v27 = 512;
                    goto LABEL_48;
                  case 38:
                    v27 = 1024;
                    goto LABEL_48;
                  case 39:
                    v27 = 0x8000000000LL;
                    goto LABEL_48;
                  case 40:
                    v27 = 0x20000000;
                    goto LABEL_48;
                  case 41:
                    v28 = 2;
                    goto LABEL_53;
                  case 43:
                    v27 = 0x100000000LL;
                    goto LABEL_48;
                  case 45:
                    v27 = 0x2000000000LL;
                    goto LABEL_48;
                  case 46:
                    v27 = 0x1000000000LL;
                    goto LABEL_48;
                  case 49:
                    if ( (v24 & 0x1C000000) != 0 )
                      sub_1560C20(v45, 1LL << ((unsigned __int8)(((unsigned int)v24 & 0x1C000000) >> 26) - 1));
                    continue;
                  case 50:
                    v28 = 0x4000;
                    goto LABEL_53;
                  case 51:
                    v28 = 0x8000;
                    goto LABEL_53;
                  case 52:
                    v28 = 0x800000000LL;
                    goto LABEL_53;
                  case 54:
                    v28 = 16;
LABEL_53:
                    if ( (v28 & v24) != 0 )
                      goto LABEL_49;
                    continue;
                  case 57:
                    v27 = 0x40000000;
LABEL_48:
                    if ( (v24 & v27) == 0 )
                      continue;
LABEL_49:
                    sub_15606E0(v45, (unsigned int)(i - 1));
                    break;
                }
              }
              if ( i == 59 )
                break;
            }
          }
LABEL_88:
          v29 = sub_1560CD0(*(_QWORD *)(v36 + 432), *(_QWORD *)&v56[8 * v21], v45);
          v30 = (unsigned int)v43;
          if ( (unsigned int)v43 >= HIDWORD(v43) )
          {
            sub_16CD150(&v42, v44, 0, 8);
            v30 = (unsigned int)v43;
          }
          v21 += 2;
          *(_QWORD *)&v42[8 * v30] = v29;
          LODWORD(v43) = v43 + 1;
          sub_14EA930(v47);
        }
        while ( v21 != v39 );
        v2 = v36;
      }
      v20 = (unsigned int)v43;
LABEL_93:
      v31 = sub_1563520(*(_QWORD *)(v2 + 432), v42, v20);
      v32 = *(char **)(v2 + 1304);
      v45[0] = v31;
      if ( v32 == *(char **)(v2 + 1312) )
      {
        sub_129A5D0((char **)(v2 + 1296), v32, v45);
      }
      else
      {
        if ( v32 )
        {
          *(_QWORD *)v32 = v31;
          v32 = *(char **)(v2 + 1304);
        }
        *(_QWORD *)(v2 + 1304) = v32 + 8;
      }
      LODWORD(v43) = 0;
    }
    else if ( v6 == 2 )
    {
      if ( (_DWORD)v57 )
      {
        v38 = v2;
        v7 = 0;
        v34 = 8LL * (unsigned int)v57;
        v8 = v2 + 1328;
        v33 = (_QWORD *)(v2 + 1320);
        do
        {
          v9 = v8;
          v10 = *(_QWORD *)&v56[v7];
          v11 = *(_QWORD *)(v38 + 1336);
          if ( !v11 )
            goto LABEL_20;
          do
          {
            while ( 1 )
            {
              v12 = *(_QWORD *)(v11 + 16);
              v13 = *(_QWORD *)(v11 + 24);
              if ( (unsigned int)*(_QWORD *)&v56[v7] <= *(_DWORD *)(v11 + 32) )
                break;
              v11 = *(_QWORD *)(v11 + 24);
              if ( !v13 )
                goto LABEL_18;
            }
            v9 = v11;
            v11 = *(_QWORD *)(v11 + 16);
          }
          while ( v12 );
LABEL_18:
          if ( v8 == v9 || (unsigned int)v10 < *(_DWORD *)(v9 + 32) )
          {
LABEL_20:
            v35 = v9;
            v14 = sub_22077B0(48);
            *(_DWORD *)(v14 + 32) = v10;
            v9 = v14;
            *(_QWORD *)(v14 + 40) = 0;
            v15 = sub_14F64E0(v33, v35, (unsigned int *)(v14 + 32));
            if ( v16 )
            {
              v17 = v15 || v8 == v16 || (unsigned int)v10 < *(_DWORD *)(v16 + 32);
              sub_220F040(v17, v9, v16, v8);
              ++*(_QWORD *)(v38 + 1360);
            }
            else
            {
              v37 = v15;
              j_j___libc_free_0(v9, 48);
              v9 = v37;
            }
          }
          v18 = (unsigned int)v43;
          if ( (unsigned int)v43 >= HIDWORD(v43) )
          {
            sub_16CD150(&v42, v44, 0, 8);
            v18 = (unsigned int)v43;
          }
          v7 += 8;
          *(_QWORD *)&v42[8 * v18] = *(_QWORD *)(v9 + 40);
          v19 = v43 + 1;
          LODWORD(v43) = v43 + 1;
        }
        while ( v34 != v7 );
        v2 = v38;
        v20 = v19;
      }
      else
      {
        v20 = (unsigned int)v43;
      }
      goto LABEL_93;
    }
  }
  *a1 = 1;
LABEL_42:
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
  if ( v56 != &v58 )
    _libc_free((unsigned __int64)v56);
  return a1;
}
