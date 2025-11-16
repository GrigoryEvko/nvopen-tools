// Function: sub_14F65E0
// Address: 0x14f65e0
//
__int64 *__fastcall sub_14F65E0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r15
  __int64 v3; // r13
  __int64 v4; // r12
  char *v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  int v8; // r13d
  unsigned int v9; // r14d
  __int64 v10; // rax
  char *v11; // r15
  __int64 v12; // r11
  __int64 v13; // rax
  __int64 *v14; // rdx
  unsigned int v15; // eax
  unsigned __int64 v16; // rcx
  __int64 v17; // r12
  __int64 v18; // rdx
  char *v19; // r8
  __int64 v20; // rbx
  unsigned int v21; // r9d
  const char *v22; // rax
  const char *v24; // rax
  __int64 v25; // r8
  unsigned int v26; // r12d
  unsigned __int64 v27; // rax
  __int64 v28; // r9
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r8
  _BOOL8 v36; // rdi
  __int64 v37; // [rsp+8h] [rbp-388h]
  unsigned int v38; // [rsp+10h] [rbp-380h]
  __int64 v39; // [rsp+18h] [rbp-378h]
  char *v40; // [rsp+20h] [rbp-370h]
  __int64 *v41; // [rsp+28h] [rbp-368h]
  unsigned int v42; // [rsp+30h] [rbp-360h]
  __int64 v43; // [rsp+30h] [rbp-360h]
  char *v44; // [rsp+38h] [rbp-358h]
  __int64 v45; // [rsp+38h] [rbp-358h]
  __int64 v46; // [rsp+38h] [rbp-358h]
  __int64 v47; // [rsp+40h] [rbp-350h]
  __int64 v48; // [rsp+40h] [rbp-350h]
  __int64 v49; // [rsp+40h] [rbp-350h]
  char *v50; // [rsp+50h] [rbp-340h] BYREF
  __int64 v51; // [rsp+58h] [rbp-338h]
  char v52; // [rsp+60h] [rbp-330h] BYREF
  unsigned __int64 v53; // [rsp+A0h] [rbp-2F0h] BYREF
  __int64 v54; // [rsp+A8h] [rbp-2E8h]
  _BYTE v55[64]; // [rsp+B0h] [rbp-2E0h] BYREF
  _QWORD v56[2]; // [rsp+F0h] [rbp-2A0h] BYREF
  int v57; // [rsp+100h] [rbp-290h] BYREF
  _QWORD *v58; // [rsp+108h] [rbp-288h]
  int *v59; // [rsp+110h] [rbp-280h]
  int *v60; // [rsp+118h] [rbp-278h]
  __int64 v61; // [rsp+120h] [rbp-270h]
  __int64 v62; // [rsp+128h] [rbp-268h]
  __int64 v63; // [rsp+130h] [rbp-260h]
  __int64 v64; // [rsp+138h] [rbp-258h]
  __int64 v65; // [rsp+140h] [rbp-250h]
  __int64 v66; // [rsp+148h] [rbp-248h]
  char *v67; // [rsp+150h] [rbp-240h] BYREF
  __int64 v68; // [rsp+158h] [rbp-238h]
  char v69; // [rsp+160h] [rbp-230h] BYREF
  char v70; // [rsp+161h] [rbp-22Fh]

  v2 = a1;
  v3 = a2 + 32;
  v4 = a2;
  if ( (unsigned __int8)sub_15127D0(a2 + 32, 10, 0) )
  {
    v70 = 1;
    v22 = "Invalid record";
    goto LABEL_20;
  }
  if ( *(_QWORD *)(a2 + 1360) )
  {
    v70 = 1;
    v22 = "Invalid multiple blocks";
LABEL_20:
    v67 = (char *)v22;
    v69 = 3;
    sub_14EE4B0(a1, a2 + 8, (__int64)&v67);
    return v2;
  }
  v5 = &v69;
  v68 = 0x4000000000LL;
  v67 = &v69;
  while ( 1 )
  {
    v6 = sub_14ED070(v3, 0);
    if ( (_DWORD)v6 == 1 )
      break;
    if ( (v6 & 0xFFFFFFFD) == 0 )
    {
      BYTE1(v57) = 1;
      v24 = "Malformed block";
LABEL_24:
      v56[0] = v24;
      LOBYTE(v57) = 3;
      sub_14EE4B0(v2, v4 + 8, (__int64)v56);
      goto LABEL_25;
    }
    LODWORD(v68) = 0;
    if ( (unsigned int)sub_1510D70(v3, HIDWORD(v6), &v67, 0) == 3 )
    {
      if ( (unsigned int)v68 <= 2 )
      {
        BYTE1(v57) = 1;
        v24 = "Invalid record";
        goto LABEL_24;
      }
      v7 = (unsigned __int64)v67;
      v47 = v4;
      v37 = v3;
      v8 = v68;
      v41 = v2;
      v40 = v5;
      v9 = 2;
      v39 = *(_QWORD *)v67;
      v10 = *((_QWORD *)v67 + 1);
      v56[0] = 0;
      v38 = v10;
      v57 = 0;
      v58 = 0;
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v64 = 0;
      v65 = 0;
      v66 = 0;
      v59 = &v57;
      v60 = &v57;
      v11 = &v52;
      while ( 1 )
      {
        v12 = *(_QWORD *)(v7 + 8LL * v9);
        v13 = v9 + 1;
        v14 = (__int64 *)(v7 + 8 * v13);
        if ( v12 )
        {
          if ( v12 == 1 )
          {
            sub_14EE720((__int64 *)&v53, v47, *v14, &v50);
            v27 = v53 & 0xFFFFFFFFFFFFFFFELL;
            if ( (v53 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            {
LABEL_38:
              v2 = v41;
              v53 = 0;
              v5 = v40;
              *v41 = v27 | 1;
              sub_14ECA90((__int64 *)&v53);
              sub_14EA930(v58);
              goto LABEL_25;
            }
            switch ( (_DWORD)v50 )
            {
              case 1:
                v26 = v9 + 3;
                sub_1560C00(v56, *(_QWORD *)&v67[8 * v9 + 16]);
                break;
              case 0x30:
                v26 = v9 + 3;
                sub_1560C20(v56, *(_QWORD *)&v67[8 * v9 + 16]);
                break;
              case 9:
                v26 = v9 + 3;
                sub_1560C40(v56, *(_QWORD *)&v67[8 * v9 + 16]);
                break;
              case 0xA:
                v26 = v9 + 3;
                sub_1560C60(v56, *(_QWORD *)&v67[8 * v9 + 16]);
                break;
              default:
                v26 = v9 + 2;
                if ( (_DWORD)v50 == 2 )
                {
                  v28 = v26;
                  v26 = v9 + 3;
                  sub_1560C80(v56, *(_QWORD *)&v67[8 * v28]);
                }
                break;
            }
          }
          else
          {
            v50 = v11;
            v51 = 0x4000000000LL;
            v53 = (unsigned __int64)v55;
            v54 = 0x4000000000LL;
            if ( !*v14 || v8 == (_DWORD)v13 )
            {
              v25 = 0;
              v26 = v9 + 2;
            }
            else
            {
              v15 = v9 + 2;
              v16 = (unsigned __int64)v11;
              v17 = *v14;
              v18 = 0;
              v19 = v11;
              v20 = v12;
              while ( 1 )
              {
                *(_BYTE *)(v16 + v18) = v17;
                v21 = v15 + 1;
                v7 = (unsigned __int64)v67;
                v18 = (unsigned int)(v51 + 1);
                LODWORD(v51) = v51 + 1;
                v17 = *(_QWORD *)&v67[8 * v15];
                if ( !v17 || v8 == v15 )
                  break;
                if ( HIDWORD(v51) <= (unsigned int)v18 )
                {
                  v42 = v15 + 1;
                  v44 = v19;
                  sub_16CD150(&v50, v19, 0, 1);
                  v18 = (unsigned int)v51;
                  v21 = v42;
                  v19 = v44;
                }
                v16 = (unsigned __int64)v50;
                v15 = v21;
              }
              v12 = v20;
              v11 = v19;
              v25 = (unsigned int)v54;
              v26 = v15 + 1;
            }
            if ( v12 == 4 )
            {
              v29 = *(_QWORD *)(v7 + 8LL * v26);
              if ( v29 && v8 != v26 )
              {
                do
                {
                  ++v26;
                  if ( HIDWORD(v54) <= (unsigned int)v25 )
                  {
                    sub_16CD150(&v53, v55, 0, 1);
                    v25 = (unsigned int)v54;
                  }
                  *(_BYTE *)(v53 + v25) = v29;
                  v25 = (unsigned int)(v54 + 1);
                  LODWORD(v54) = v54 + 1;
                  v29 = *(_QWORD *)&v67[8 * v26];
                }
                while ( v29 && v26 != v8 );
                ++v26;
              }
              else
              {
                ++v26;
              }
            }
            sub_1562A10(v56, v50, (unsigned int)v51, v53, v25);
            if ( (_BYTE *)v53 != v55 )
              _libc_free(v53);
            if ( v50 != v11 )
              _libc_free((unsigned __int64)v50);
          }
        }
        else
        {
          sub_14EE720((__int64 *)&v53, v47, *v14, &v50);
          v27 = v53 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v53 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_38;
          v26 = v9 + 2;
          sub_15606E0(v56, (unsigned int)v50);
        }
        v9 = v26;
        if ( v26 == v8 )
          break;
        v7 = (unsigned __int64)v67;
      }
      v4 = v47;
      v3 = v37;
      v2 = v41;
      v5 = v40;
      v30 = *(_QWORD *)(v47 + 1336);
      v31 = *(_QWORD *)(v47 + 432);
      v32 = v47 + 1328;
      if ( !v30 )
        goto LABEL_65;
      do
      {
        if ( (unsigned int)v39 > *(_DWORD *)(v30 + 32) )
        {
          v30 = *(_QWORD *)(v30 + 24);
        }
        else
        {
          v32 = v30;
          v30 = *(_QWORD *)(v30 + 16);
        }
      }
      while ( v30 );
      if ( v32 == v47 + 1328 || (unsigned int)v39 < *(_DWORD *)(v32 + 32) )
      {
LABEL_65:
        v45 = *(_QWORD *)(v47 + 432);
        v43 = v47 + 1328;
        v48 = v32;
        v32 = sub_22077B0(48);
        *(_QWORD *)(v32 + 40) = 0;
        *(_DWORD *)(v32 + 32) = v39;
        v33 = sub_14F64E0((_QWORD *)(v4 + 1320), v48, (unsigned int *)(v32 + 32));
        v35 = v45;
        if ( v34 )
        {
          v36 = v43 == v34 || v33 || (unsigned int)v39 < *(_DWORD *)(v34 + 32);
          sub_220F040(v36, v32, v34, v43);
          v31 = v45;
          ++*(_QWORD *)(v4 + 1360);
        }
        else
        {
          v46 = v33;
          v49 = v35;
          j_j___libc_free_0(v32, 48);
          v31 = v49;
          v32 = v46;
        }
      }
      *(_QWORD *)(v32 + 40) = sub_1560CD0(v31, v38, v56);
      sub_14EA930(v58);
    }
  }
  *v2 = 1;
LABEL_25:
  if ( v67 != v5 )
    _libc_free((unsigned __int64)v67);
  return v2;
}
