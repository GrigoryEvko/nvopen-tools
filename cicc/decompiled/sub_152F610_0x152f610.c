// Function: sub_152F610
// Address: 0x152f610
//
void __fastcall sub_152F610(_DWORD **a1)
{
  unsigned int *v1; // rsi
  int i; // edi
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // rcx
  _DWORD *v7; // r8
  int v8; // r11d
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rsi
  int j; // eax
  _DWORD *v12; // rsi
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // rax
  signed __int64 v24; // rdx
  __int64 v25; // rax
  signed __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r11
  __int64 v29; // rdx
  __int64 v30; // r10
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  _BYTE *v34; // rcx
  __int64 k; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  _BYTE *v38; // rcx
  __int64 m; // rax
  __int64 v40; // rax
  __int64 v41; // [rsp+0h] [rbp-2A0h]
  unsigned int *v42; // [rsp+10h] [rbp-290h]
  __int64 v43; // [rsp+18h] [rbp-288h]
  __int64 v44; // [rsp+18h] [rbp-288h]
  __int64 v45; // [rsp+20h] [rbp-280h]
  signed __int64 v46; // [rsp+20h] [rbp-280h]
  __int64 v48; // [rsp+38h] [rbp-268h]
  __int64 v49; // [rsp+38h] [rbp-268h]
  __int64 v50; // [rsp+38h] [rbp-268h]
  __int64 v51; // [rsp+38h] [rbp-268h]
  __int64 v52; // [rsp+40h] [rbp-260h]
  signed __int64 v53; // [rsp+40h] [rbp-260h]
  signed __int64 v54; // [rsp+40h] [rbp-260h]
  __int64 v55; // [rsp+40h] [rbp-260h]
  __int64 v56; // [rsp+40h] [rbp-260h]
  __int64 v57; // [rsp+40h] [rbp-260h]
  __int64 v58; // [rsp+40h] [rbp-260h]
  unsigned int *v59; // [rsp+48h] [rbp-258h]
  __int64 v60; // [rsp+50h] [rbp-250h] BYREF
  __int64 v61; // [rsp+58h] [rbp-248h] BYREF
  _BYTE *v62; // [rsp+60h] [rbp-240h] BYREF
  __int64 v63; // [rsp+68h] [rbp-238h]
  _BYTE v64[560]; // [rsp+70h] [rbp-230h] BYREF

  if ( a1[48] != a1[49] )
  {
    sub_1526BE0(*a1, 0xAu, 3u);
    v1 = a1[49];
    v62 = v64;
    v63 = 0x4000000000LL;
    v42 = v1;
    if ( a1[48] != v1 )
    {
      v59 = a1[48];
      for ( i = 64; ; i = HIDWORD(v63) )
      {
        v3 = 0;
        v4 = *((_QWORD *)v59 + 1);
        v5 = *v59;
        v60 = v4;
        if ( v4 )
        {
          v6 = *((unsigned int *)a1 + 94);
          v7 = a1[45];
          if ( (_DWORD)v6 )
          {
            v8 = 1;
            v9 = (((((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4) | ((unsigned __int64)(unsigned int)(37 * v5) << 32))
                 - 1
                 - ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32)) >> 22)
               ^ ((((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4) | ((unsigned __int64)(unsigned int)(37 * v5) << 32))
                - 1
                - ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32));
            v10 = ((9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13)))) >> 15)
                ^ (9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13))));
            for ( j = (v6 - 1) & (((v10 - 1 - (v10 << 27)) >> 31) ^ (v10 - 1 - ((_DWORD)v10 << 27))); ; j = (v6 - 1) & v13 )
            {
              v12 = &v7[6 * j];
              if ( (_DWORD)v5 == *v12 && v4 == *((_QWORD *)v12 + 1) )
                break;
              if ( *v12 == -1 && *((_QWORD *)v12 + 1) == -4 )
                goto LABEL_11;
              v13 = v8 + j;
              ++v8;
            }
          }
          else
          {
LABEL_11:
            v12 = &v7[6 * v6];
          }
          v3 = (unsigned int)v12[4];
        }
        v14 = 0;
        if ( !i )
        {
          sub_16CD150(&v62, v64, 0, 8);
          v14 = (unsigned int)v63;
        }
        *(_QWORD *)&v62[8 * v14] = v3;
        v15 = (unsigned int)(v63 + 1);
        LODWORD(v63) = v15;
        if ( HIDWORD(v63) <= (unsigned int)v15 )
        {
          sub_16CD150(&v62, v64, 0, 8);
          v15 = (unsigned int)v63;
        }
        *(_QWORD *)&v62[8 * v15] = v5;
        LODWORD(v63) = v63 + 1;
        v16 = (__int64 *)sub_155EE30(&v60);
        v17 = sub_155EE40(&v60);
        if ( v16 != (__int64 *)v17 )
          break;
LABEL_52:
        sub_152F3D0(*a1, 3u, (__int64)&v62, 0);
        v59 += 4;
        LODWORD(v63) = 0;
        if ( v42 == v59 )
          goto LABEL_54;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v61 = *v16;
          if ( !(unsigned __int8)sub_155D3A0(&v61) )
            break;
          v18 = (unsigned int)v63;
          if ( (unsigned int)v63 >= HIDWORD(v63) )
          {
            sub_16CD150(&v62, v64, 0, 8);
            v18 = (unsigned int)v63;
          }
          *(_QWORD *)&v62[8 * v18] = 0;
          LODWORD(v63) = v63 + 1;
          v19 = byte_4292D20[(unsigned int)sub_155D410(&v61) - 1];
          v20 = (unsigned int)v63;
          if ( (unsigned int)v63 < HIDWORD(v63) )
            goto LABEL_22;
LABEL_31:
          v52 = v19;
          sub_16CD150(&v62, v64, 0, 8);
          v20 = (unsigned int)v63;
          v19 = v52;
LABEL_22:
          *(_QWORD *)&v62[8 * v20] = v19;
          LODWORD(v63) = v63 + 1;
LABEL_23:
          if ( (__int64 *)v17 == ++v16 )
            goto LABEL_52;
        }
        if ( (unsigned __int8)sub_155D3C0(&v61) )
        {
          v21 = (unsigned int)v63;
          if ( (unsigned int)v63 >= HIDWORD(v63) )
          {
            sub_16CD150(&v62, v64, 0, 8);
            v21 = (unsigned int)v63;
          }
          *(_QWORD *)&v62[8 * v21] = 1;
          LODWORD(v63) = v63 + 1;
          v22 = byte_4292D20[(unsigned int)sub_155D410(&v61) - 1];
          v23 = (unsigned int)v63;
          if ( (unsigned int)v63 >= HIDWORD(v63) )
          {
            v57 = v22;
            sub_16CD150(&v62, v64, 0, 8);
            v23 = (unsigned int)v63;
            v22 = v57;
          }
          *(_QWORD *)&v62[8 * v23] = v22;
          LODWORD(v63) = v63 + 1;
          v19 = sub_155D4B0(&v61);
          v20 = (unsigned int)v63;
          if ( (unsigned int)v63 < HIDWORD(v63) )
            goto LABEL_22;
          goto LABEL_31;
        }
        v48 = sub_155D7D0(&v61);
        v53 = v24;
        v25 = sub_155D8B0(&v61);
        v26 = v53;
        v27 = v48;
        v28 = v25;
        v30 = v29;
        v31 = 3 - ((v29 == 0) - 1LL);
        v32 = (unsigned int)v63;
        if ( (unsigned int)v63 >= HIDWORD(v63) )
        {
          v41 = v30;
          v44 = v28;
          v46 = v53;
          v58 = v31;
          sub_16CD150(&v62, v64, 0, 8);
          v32 = (unsigned int)v63;
          v30 = v41;
          v28 = v44;
          v26 = v46;
          v27 = v48;
          v31 = v58;
        }
        *(_QWORD *)&v62[8 * v32] = v31;
        LODWORD(v63) = v63 + 1;
        v33 = (unsigned int)v63;
        if ( v26 > HIDWORD(v63) - (unsigned __int64)(unsigned int)v63 )
        {
          v43 = v30;
          v45 = v28;
          v49 = v27;
          v54 = v26;
          sub_16CD150(&v62, v64, v26 + (unsigned int)v63, 8);
          v33 = (unsigned int)v63;
          v30 = v43;
          v28 = v45;
          v27 = v49;
          v26 = v54;
        }
        v34 = &v62[8 * v33];
        if ( v26 > 0 )
        {
          for ( k = 0; k != v26; ++k )
            *(_QWORD *)&v34[8 * k] = *(char *)(v27 + k);
          LODWORD(v33) = v63;
        }
        LODWORD(v63) = v33 + v26;
        v36 = (unsigned int)(v33 + v26);
        if ( HIDWORD(v63) <= (unsigned int)(v33 + v26) )
        {
          v51 = v30;
          v56 = v28;
          sub_16CD150(&v62, v64, 0, 8);
          v36 = (unsigned int)v63;
          v30 = v51;
          v28 = v56;
        }
        *(_QWORD *)&v62[8 * v36] = 0;
        v37 = (unsigned int)(v63 + 1);
        LODWORD(v63) = v63 + 1;
        if ( !v30 )
          goto LABEL_23;
        if ( v30 > (unsigned __int64)HIDWORD(v63) - v37 )
        {
          v50 = v28;
          v55 = v30;
          sub_16CD150(&v62, v64, v30 + v37, 8);
          v37 = (unsigned int)v63;
          v28 = v50;
          v30 = v55;
        }
        v38 = &v62[8 * v37];
        if ( v30 > 0 )
        {
          for ( m = 0; m != v30; ++m )
            *(_QWORD *)&v38[8 * m] = *(char *)(v28 + m);
          LODWORD(v37) = v63;
        }
        LODWORD(v63) = v37 + v30;
        v40 = (unsigned int)(v37 + v30);
        if ( HIDWORD(v63) <= (unsigned int)v63 )
        {
          sub_16CD150(&v62, v64, 0, 8);
          v40 = (unsigned int)v63;
        }
        ++v16;
        *(_QWORD *)&v62[8 * v40] = 0;
        LODWORD(v63) = v63 + 1;
        if ( (__int64 *)v17 == v16 )
          goto LABEL_52;
      }
    }
LABEL_54:
    sub_15263C0((__int64 **)*a1);
    if ( v62 != v64 )
      _libc_free((unsigned __int64)v62);
  }
}
