// Function: sub_31A5620
// Address: 0x31a5620
//
__int64 __fastcall sub_31A5620(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  bool v12; // al
  __int64 v13; // r14
  int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // r14
  bool v17; // al
  _BYTE *v18; // r8
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // r14
  bool v22; // al
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r14
  bool v26; // al
  int v27; // eax
  __int64 v28; // rcx
  bool v29; // al
  bool v31; // al
  bool v32; // al
  bool v33; // al
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rsi
  signed __int64 v37; // rax
  __int64 v38; // rcx
  int v39; // eax
  __int64 v40; // rcx
  __int64 v41; // r12
  __int64 v42; // rdx
  _BYTE *v43; // r14
  int v44; // eax
  __int64 v45; // rcx
  __int64 v46; // r12
  _BYTE *v47; // r14
  int v48; // eax
  __int64 v49; // r12
  __int64 v50; // rcx
  _BYTE *v51; // r14
  __int64 v52; // [rsp+0h] [rbp-A0h]
  _BYTE *v53; // [rsp+8h] [rbp-98h]
  _BYTE *v54; // [rsp+8h] [rbp-98h]
  _BYTE *v55; // [rsp+8h] [rbp-98h]
  _BYTE *v56; // [rsp+8h] [rbp-98h]
  __int64 v57; // [rsp+10h] [rbp-90h]
  unsigned __int8 v58; // [rsp+1Fh] [rbp-81h]
  unsigned __int64 v59; // [rsp+20h] [rbp-80h] BYREF
  __int64 v60; // [rsp+28h] [rbp-78h]
  __int64 v61; // [rsp+30h] [rbp-70h]
  int v62; // [rsp+38h] [rbp-68h]
  __int64 v63; // [rsp+40h] [rbp-60h]
  __int64 v64; // [rsp+48h] [rbp-58h]
  _BYTE *v65; // [rsp+50h] [rbp-50h] BYREF
  __int64 v66; // [rsp+58h] [rbp-48h]
  _BYTE v67[64]; // [rsp+60h] [rbp-40h] BYREF

  if ( !**(_QWORD **)(a1 + 408) )
    return 1;
  v58 = sub_31A4BE0(*(_QWORD *)(a1 + 416), a2, a3, a4, a5, a6);
  if ( v58 )
    return 1;
  if ( (_BYTE)a2 )
  {
    v8 = *(_QWORD *)(a1 + 160);
    v9 = 88LL * *(unsigned int *)(a1 + 168);
    v10 = v8 + v9;
    v11 = 0x2E8BA2E8BA2E8BA3LL * (v9 >> 3);
    v52 = v10;
    if ( v11 >> 2 )
    {
      v57 = v8 + 352 * (v11 >> 2);
      while ( 1 )
      {
        v59 = 6;
        v60 = 0;
        v61 = *(_QWORD *)(v8 + 24);
        if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
          sub_BD6050(&v59, *(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        v27 = *(_DWORD *)(v8 + 32);
        v62 = v27;
        v28 = *(_QWORD *)(v8 + 40);
        v63 = v28;
        v13 = *(_QWORD *)(v8 + 48);
        v65 = v67;
        v64 = v13;
        v66 = 0x200000000LL;
        if ( *(_DWORD *)(v8 + 64) )
          break;
        if ( v27 != 3 )
        {
          if ( v61 == 0 || v61 == -4096 )
            goto LABEL_16;
          v13 = 0;
          if ( v61 == -8192 )
            goto LABEL_16;
          goto LABEL_14;
        }
        if ( !v13 )
        {
          if ( v61 == 0 || v61 == -4096 || v61 == -8192 )
            goto LABEL_16;
LABEL_14:
          sub_BD60C0(&v59);
          goto LABEL_15;
        }
        v12 = sub_B451B0(v13);
        v6 = (unsigned __int64)v67;
        if ( v12 )
          goto LABEL_9;
LABEL_12:
        if ( v61 != -4096 && v61 != 0 && v61 != -8192 )
          goto LABEL_14;
LABEL_15:
        if ( v13 )
          goto LABEL_98;
LABEL_16:
        v59 = 6;
        v60 = 0;
        v61 = *(_QWORD *)(v8 + 112);
        if ( v61 != -4096 && v61 != 0 && v61 != -8192 )
          sub_BD6050(&v59, *(_QWORD *)(v8 + 96) & 0xFFFFFFFFFFFFFFF8LL);
        v14 = *(_DWORD *)(v8 + 120);
        v62 = v14;
        v15 = *(_QWORD *)(v8 + 128);
        v63 = v15;
        v16 = *(_QWORD *)(v8 + 136);
        v65 = v67;
        v64 = v16;
        v66 = 0x200000000LL;
        v7 = *(unsigned int *)(v8 + 152);
        if ( (_DWORD)v7 )
        {
          sub_31A3C40((__int64)&v65, v8 + 144, v10, v15, v6, v7);
          if ( v62 == 3 )
          {
            v16 = v64;
            v18 = v65;
            if ( v64 )
            {
              v54 = v65;
              v31 = sub_B451B0(v64);
              v18 = v54;
              if ( v31 )
LABEL_23:
                v16 = 0;
            }
          }
          else
          {
            v18 = v65;
            v16 = 0;
          }
          if ( v18 != v67 )
            _libc_free((unsigned __int64)v18);
          goto LABEL_26;
        }
        if ( v14 != 3 )
        {
          if ( v61 == -4096 || v61 == 0 )
            goto LABEL_30;
          v16 = 0;
          if ( v61 == -8192 )
            goto LABEL_30;
          goto LABEL_28;
        }
        if ( !v16 )
        {
          if ( v61 == 0 || v61 == -4096 || v61 == -8192 )
            goto LABEL_30;
LABEL_28:
          sub_BD60C0(&v59);
          goto LABEL_29;
        }
        v17 = sub_B451B0(v16);
        v18 = v67;
        if ( v17 )
          goto LABEL_23;
LABEL_26:
        if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
          goto LABEL_28;
LABEL_29:
        if ( v16 )
        {
          v8 += 88;
          goto LABEL_98;
        }
LABEL_30:
        v59 = 6;
        v60 = 0;
        v61 = *(_QWORD *)(v8 + 200);
        if ( v61 != -4096 && v61 != 0 && v61 != -8192 )
          sub_BD6050(&v59, *(_QWORD *)(v8 + 184) & 0xFFFFFFFFFFFFFFF8LL);
        v19 = *(_DWORD *)(v8 + 208);
        v62 = v19;
        v20 = *(_QWORD *)(v8 + 216);
        v63 = v20;
        v21 = *(_QWORD *)(v8 + 224);
        v65 = v67;
        v64 = v21;
        v66 = 0x200000000LL;
        v6 = *(unsigned int *)(v8 + 240);
        if ( (_DWORD)v6 )
        {
          sub_31A3C40((__int64)&v65, v8 + 232, v10, v20, v6, v7);
          if ( v62 == 3 )
          {
            v21 = v64;
            v6 = (unsigned __int64)v65;
            if ( v64 )
            {
              v55 = v65;
              v32 = sub_B451B0(v64);
              v6 = (unsigned __int64)v55;
              if ( v32 )
LABEL_37:
                v21 = 0;
            }
          }
          else
          {
            v6 = (unsigned __int64)v65;
            v21 = 0;
          }
          if ( (_BYTE *)v6 != v67 )
            _libc_free(v6);
          goto LABEL_40;
        }
        if ( v19 != 3 )
        {
          if ( v61 == -4096 || v61 == 0 )
            goto LABEL_44;
          v21 = 0;
          if ( v61 == -8192 )
            goto LABEL_44;
          goto LABEL_42;
        }
        if ( !v21 )
        {
          if ( v61 == 0 || v61 == -4096 || v61 == -8192 )
            goto LABEL_44;
LABEL_42:
          sub_BD60C0(&v59);
          goto LABEL_43;
        }
        v22 = sub_B451B0(v21);
        v6 = (unsigned __int64)v67;
        if ( v22 )
          goto LABEL_37;
LABEL_40:
        if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
          goto LABEL_42;
LABEL_43:
        if ( v21 )
        {
          v8 += 176;
          goto LABEL_98;
        }
LABEL_44:
        v59 = 6;
        v60 = 0;
        v61 = *(_QWORD *)(v8 + 288);
        if ( v61 != -4096 && v61 != 0 && v61 != -8192 )
          sub_BD6050(&v59, *(_QWORD *)(v8 + 272) & 0xFFFFFFFFFFFFFFF8LL);
        v23 = *(_DWORD *)(v8 + 296);
        v62 = v23;
        v24 = *(_QWORD *)(v8 + 304);
        v63 = v24;
        v25 = *(_QWORD *)(v8 + 312);
        v65 = v67;
        v64 = v25;
        v66 = 0x200000000LL;
        if ( *(_DWORD *)(v8 + 328) )
        {
          sub_31A3C40((__int64)&v65, v8 + 320, v10, v24, v6, v7);
          if ( v62 == 3 )
          {
            v25 = v64;
            v6 = (unsigned __int64)v65;
            if ( v64 )
            {
              v56 = v65;
              v33 = sub_B451B0(v64);
              v6 = (unsigned __int64)v56;
              if ( v33 )
LABEL_51:
                v25 = 0;
            }
          }
          else
          {
            v6 = (unsigned __int64)v65;
            v25 = 0;
          }
          if ( (_BYTE *)v6 != v67 )
            _libc_free(v6);
          goto LABEL_54;
        }
        if ( v23 != 3 )
        {
          if ( v61 == -4096 || v61 == 0 )
            goto LABEL_58;
          v25 = 0;
          if ( v61 == -8192 )
            goto LABEL_58;
LABEL_56:
          sub_BD60C0(&v59);
          goto LABEL_57;
        }
        if ( !v25 )
        {
          if ( v61 == 0 || v61 == -4096 || v61 == -8192 )
            goto LABEL_58;
          goto LABEL_56;
        }
        v26 = sub_B451B0(v25);
        v6 = (unsigned __int64)v67;
        if ( v26 )
          goto LABEL_51;
LABEL_54:
        if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
          goto LABEL_56;
LABEL_57:
        if ( v25 )
        {
          v8 += 264;
          goto LABEL_98;
        }
LABEL_58:
        v8 += 352;
        if ( v57 == v8 )
        {
          v11 = 0x2E8BA2E8BA2E8BA3LL * ((v52 - v8) >> 3);
          goto LABEL_126;
        }
      }
      sub_31A3C40((__int64)&v65, v8 + 56, v10, v28, v6, v7);
      if ( v62 == 3 )
      {
        v13 = v64;
        v6 = (unsigned __int64)v65;
        if ( v64 )
        {
          v53 = v65;
          v29 = sub_B451B0(v64);
          v6 = (unsigned __int64)v53;
          if ( v29 )
LABEL_9:
            v13 = 0;
        }
      }
      else
      {
        v6 = (unsigned __int64)v65;
        v13 = 0;
      }
      if ( (_BYTE *)v6 != v67 )
        _libc_free(v6);
      goto LABEL_12;
    }
LABEL_126:
    if ( v11 == 2 )
    {
LABEL_166:
      v59 = 6;
      v60 = 0;
      v61 = *(_QWORD *)(v8 + 24);
      if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
        sub_BD6050(&v59, *(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      v48 = *(_DWORD *)(v8 + 32);
      v62 = v48;
      v63 = *(_QWORD *)(v8 + 40);
      v49 = *(_QWORD *)(v8 + 48);
      v65 = v67;
      v64 = v49;
      v66 = 0x200000000LL;
      v50 = *(unsigned int *)(v8 + 64);
      if ( !(_DWORD)v50 )
      {
        if ( v48 == 3 )
        {
          if ( v49 )
          {
            v51 = v67;
            if ( sub_B451B0(v49) )
            {
              v49 = 0;
              goto LABEL_175;
            }
LABEL_173:
            if ( v51 != v67 )
              _libc_free((unsigned __int64)v51);
LABEL_175:
            if ( v61 == 0 || v61 == -4096 || v61 == -8192 )
            {
LABEL_178:
              if ( v49 )
                goto LABEL_98;
              goto LABEL_179;
            }
LABEL_177:
            sub_BD60C0(&v59);
            goto LABEL_178;
          }
          if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
            goto LABEL_177;
        }
        else if ( v61 != -4096 && v61 != 0 )
        {
          v49 = 0;
          if ( v61 != -8192 )
            goto LABEL_177;
        }
LABEL_179:
        v8 += 88;
        goto LABEL_129;
      }
      sub_31A3C40((__int64)&v65, v8 + 56, 0x200000000LL, v50, v6, v7);
      if ( v62 == 3 )
      {
        v49 = v64;
        v51 = v65;
        if ( v64 && sub_B451B0(v64) )
          v49 = 0;
      }
      else
      {
        v51 = v65;
        v49 = 0;
      }
      goto LABEL_173;
    }
    if ( v11 != 3 )
    {
      if ( v11 != 1 )
        goto LABEL_99;
LABEL_129:
      v59 = 6;
      v60 = 0;
      v61 = *(_QWORD *)(v8 + 24);
      if ( v61 != -4096 && v61 != 0 && v61 != -8192 )
        sub_BD6050(&v59, *(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      v39 = *(_DWORD *)(v8 + 32);
      v62 = v39;
      v40 = *(_QWORD *)(v8 + 40);
      v63 = v40;
      v41 = *(_QWORD *)(v8 + 48);
      v65 = v67;
      v64 = v41;
      v66 = 0x200000000LL;
      v42 = *(unsigned int *)(v8 + 64);
      if ( (_DWORD)v42 )
      {
        sub_31A3C40((__int64)&v65, v8 + 56, v42, v40, v6, v7);
        if ( v62 == 3 )
        {
          v41 = v64;
          v43 = v65;
          if ( v64 && sub_B451B0(v64) )
            v41 = 0;
        }
        else
        {
          v43 = v65;
          v41 = 0;
        }
      }
      else
      {
        if ( v39 != 3 )
        {
          if ( v61 == 0 || v61 == -4096 )
            goto LABEL_99;
          v41 = 0;
          if ( v61 == -8192 )
            goto LABEL_99;
          goto LABEL_140;
        }
        if ( !v41 )
        {
          if ( v61 == 0 || v61 == -4096 || v61 == -8192 )
            goto LABEL_99;
          goto LABEL_140;
        }
        v43 = v67;
        if ( sub_B451B0(v41) )
        {
          v41 = 0;
          goto LABEL_138;
        }
      }
      if ( v43 != v67 )
        _libc_free((unsigned __int64)v43);
LABEL_138:
      if ( v61 == -4096 || v61 == 0 || v61 == -8192 )
      {
LABEL_141:
        if ( v41 )
        {
LABEL_98:
          if ( v52 == v8 )
            goto LABEL_99;
          return v58;
        }
LABEL_99:
        v34 = *(_QWORD *)(a1 + 112);
        v35 = 184LL * *(unsigned int *)(a1 + 120);
        v36 = v34 + v35;
        v37 = 0xD37A6F4DE9BD37A7LL * (v35 >> 3);
        if ( v37 >> 2 )
        {
          v38 = v34 + 736 * (v37 >> 2);
          while ( !*(_QWORD *)(v34 + 56) || *(_BYTE *)(v34 + 73) )
          {
            if ( *(_QWORD *)(v34 + 240) && !*(_BYTE *)(v34 + 257) )
            {
              v34 += 184;
              return v36 == v34;
            }
            if ( *(_QWORD *)(v34 + 424) && !*(_BYTE *)(v34 + 441) )
            {
              v34 += 368;
              return v36 == v34;
            }
            if ( *(_QWORD *)(v34 + 608) && !*(_BYTE *)(v34 + 625) )
            {
              v34 += 552;
              return v36 == v34;
            }
            v34 += 736;
            if ( v38 == v34 )
            {
              v37 = 0xD37A6F4DE9BD37A7LL * ((v36 - v34) >> 3);
              goto LABEL_147;
            }
          }
          return v36 == v34;
        }
LABEL_147:
        if ( v37 != 2 )
        {
          if ( v37 != 3 )
          {
            if ( v37 != 1 )
            {
LABEL_150:
              v34 = v36;
              return v36 == v34;
            }
LABEL_213:
            if ( *(_QWORD *)(v34 + 56) )
            {
              if ( *(_BYTE *)(v34 + 73) )
                v34 = v36;
              return v36 == v34;
            }
            goto LABEL_150;
          }
          if ( *(_QWORD *)(v34 + 56) && !*(_BYTE *)(v34 + 73) )
            return v36 == v34;
          v34 += 184;
        }
        if ( *(_QWORD *)(v34 + 56) && !*(_BYTE *)(v34 + 73) )
          return v36 == v34;
        v34 += 184;
        goto LABEL_213;
      }
LABEL_140:
      sub_BD60C0(&v59);
      goto LABEL_141;
    }
    v59 = 6;
    v60 = 0;
    v61 = *(_QWORD *)(v8 + 24);
    if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
      sub_BD6050(&v59, *(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    v44 = *(_DWORD *)(v8 + 32);
    v62 = v44;
    v45 = *(_QWORD *)(v8 + 40);
    v63 = v45;
    v46 = *(_QWORD *)(v8 + 48);
    v65 = v67;
    v64 = v46;
    v66 = 0x200000000LL;
    if ( !*(_DWORD *)(v8 + 64) )
    {
      if ( v44 == 3 )
      {
        if ( v46 )
        {
          v47 = v67;
          if ( !sub_B451B0(v46) )
            goto LABEL_161;
          goto LABEL_158;
        }
        if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
          goto LABEL_163;
      }
      else if ( v61 != -4096 && v61 != 0 )
      {
        v46 = 0;
        if ( v61 != -8192 )
          goto LABEL_163;
      }
LABEL_165:
      v8 += 88;
      goto LABEL_166;
    }
    sub_31A3C40((__int64)&v65, v8 + 56, 0x200000000LL, v45, v6, v7);
    if ( v62 == 3 )
    {
      v46 = v64;
      v47 = v65;
      if ( v64 && sub_B451B0(v64) )
LABEL_158:
        v46 = 0;
    }
    else
    {
      v47 = v65;
      v46 = 0;
    }
    if ( v47 != v67 )
      _libc_free((unsigned __int64)v47);
LABEL_161:
    if ( v61 == 0 || v61 == -4096 || v61 == -8192 )
    {
LABEL_164:
      if ( v46 )
        goto LABEL_98;
      goto LABEL_165;
    }
LABEL_163:
    sub_BD60C0(&v59);
    goto LABEL_164;
  }
  return v58;
}
