// Function: sub_31D91D0
// Address: 0x31d91d0
//
__int64 __fastcall sub_31D91D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _BYTE *v4; // rdx
  __int64 v5; // rax
  unsigned __int8 v6; // dl
  __int64 v7; // r13
  __int64 v8; // r14
  bool v9; // al
  _QWORD *v10; // rsi
  _BYTE *v11; // rcx
  unsigned __int8 v12; // si
  __int64 v13; // rcx
  __int64 v14; // rdi
  unsigned __int8 *v15; // rax
  size_t v16; // rdx
  _QWORD *v17; // rdi
  _BYTE *v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rax
  size_t v21; // rdx
  _DWORD *v22; // rdi
  void *v23; // rsi
  unsigned __int64 v24; // rax
  size_t v25; // r13
  unsigned __int64 *v26; // r13
  unsigned __int64 *v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rax
  __int16 v30; // cx
  __int64 v31; // r14
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  _WORD *v36; // rdx
  _QWORD *v37; // rdi
  __int64 v38; // rdi
  _BYTE *v39; // rax
  __int64 v40; // rdi
  _BYTE *v41; // rax
  __int64 v42; // rdi
  void (__fastcall *v43)(__int64, unsigned __int64 **, __int64); // rax
  __int64 *v44; // rsi
  double v45; // xmm0_8
  __int64 (*v46)(); // rax
  __int64 v47; // rax
  _BYTE *v48; // rdx
  __int64 v49; // rax
  _QWORD *v50; // rdi
  _QWORD *v51; // rbx
  _DWORD *v52; // rax
  _WORD *v53; // rdx
  _QWORD *v54; // r8
  double v55; // xmm0_8
  __int64 v56; // rdx
  _BYTE *v57; // rax
  _BYTE *v58; // rax
  _QWORD *v59; // rdi
  __int64 v60; // rdi
  _BYTE *v61; // rax
  _BYTE *v62; // rax
  _QWORD *v63; // r15
  unsigned __int8 *v64; // rax
  size_t v65; // rdx
  void *v66; // rdi
  __int64 v67; // r13
  _QWORD *v68; // rdi
  _BYTE *v69; // rax
  unsigned __int64 *v70; // r13
  __int64 v71; // rax
  signed __int64 v72; // [rsp+10h] [rbp-170h]
  __int64 v73; // [rsp+10h] [rbp-170h]
  __int64 v74; // [rsp+10h] [rbp-170h]
  __int64 v75; // [rsp+10h] [rbp-170h]
  char v76; // [rsp+18h] [rbp-168h]
  unsigned __int64 *v77; // [rsp+18h] [rbp-168h]
  size_t v78; // [rsp+18h] [rbp-168h]
  size_t v80; // [rsp+28h] [rbp-158h]
  __int64 v81; // [rsp+28h] [rbp-158h]
  unsigned __int64 *v82; // [rsp+28h] [rbp-158h]
  unsigned int v83; // [rsp+3Ch] [rbp-144h] BYREF
  unsigned __int64 *v84; // [rsp+40h] [rbp-140h] BYREF
  _QWORD *v85; // [rsp+48h] [rbp-138h]
  void (__fastcall *v86)(unsigned __int64 **, unsigned __int64 **, __int64); // [rsp+50h] [rbp-130h]
  void (__fastcall *v87)(unsigned __int64 **, _QWORD *); // [rsp+58h] [rbp-128h]
  __int16 v88; // [rsp+60h] [rbp-120h]
  _QWORD v89[3]; // [rsp+70h] [rbp-110h] BYREF
  unsigned __int64 v90; // [rsp+88h] [rbp-F8h]
  void *dest; // [rsp+90h] [rbp-F0h]
  __int64 v92; // [rsp+98h] [rbp-E8h]
  unsigned __int64 **v93; // [rsp+A0h] [rbp-E0h]
  unsigned __int64 *v94; // [rsp+B0h] [rbp-D0h] BYREF
  _QWORD *v95; // [rsp+B8h] [rbp-C8h]
  __int64 v96; // [rsp+C0h] [rbp-C0h]
  _BYTE v97[184]; // [rsp+C8h] [rbp-B8h] BYREF

  v2 = a1;
  if ( *(_WORD *)(a1 + 68) == 14 && (*(_DWORD *)(a1 + 40) & 0xFFFFFF) != 4 )
    return 0;
  v94 = (unsigned __int64 *)v97;
  v92 = 0x100000000LL;
  v89[0] = &unk_49DD288;
  v93 = &v94;
  v95 = 0;
  v96 = 128;
  v89[1] = 2;
  v89[2] = 0;
  v90 = 0;
  dest = 0;
  sub_CB5980((__int64)v89, 0, 0, 0);
  v4 = dest;
  if ( v90 - (unsigned __int64)dest <= 0xC )
  {
    sub_CB6200((__int64)v89, "DEBUG_VALUE: ", 0xDu);
  }
  else
  {
    *((_DWORD *)dest + 2) = 977622348;
    *(_QWORD *)v4 = 0x41565F4755424544LL;
    v4[12] = 32;
    dest = (char *)dest + 13;
  }
  v5 = sub_2E89170(a1);
  v6 = *(_BYTE *)(v5 - 16);
  v7 = v5;
  v8 = v5 - 16;
  v9 = (v6 & 2) != 0;
  if ( (v6 & 2) == 0 )
  {
    v10 = (_QWORD *)(v8 - 8LL * ((v6 >> 2) & 0xF));
    v11 = (_BYTE *)*v10;
    if ( *(_BYTE *)*v10 != 18 )
      goto LABEL_20;
    v12 = *(v11 - 16);
    if ( (v12 & 2) != 0 )
      goto LABEL_10;
LABEL_74:
    v13 = (__int64)&v11[-8 * ((v12 >> 2) & 0xF) - 16];
    goto LABEL_11;
  }
  v10 = *(_QWORD **)(v7 - 32);
  v11 = (_BYTE *)*v10;
  if ( *(_BYTE *)*v10 != 18 )
    goto LABEL_20;
  v12 = *(v11 - 16);
  if ( (v12 & 2) == 0 )
    goto LABEL_74;
LABEL_10:
  v13 = *((_QWORD *)v11 - 4);
LABEL_11:
  v14 = *(_QWORD *)(v13 + 16);
  if ( v14 )
  {
    v15 = (unsigned __int8 *)sub_B91420(v14);
    if ( v16 )
    {
      if ( v90 - (unsigned __int64)dest < v16 )
      {
        v71 = sub_CB6200((__int64)v89, v15, v16);
        v18 = *(_BYTE **)(v71 + 32);
        v17 = (_QWORD *)v71;
      }
      else
      {
        v80 = v16;
        memcpy(dest, v15, v16);
        v17 = v89;
        v18 = (char *)dest + v80;
        dest = (char *)dest + v80;
      }
      if ( (_BYTE *)v17[3] == v18 )
      {
        sub_CB6200((__int64)v17, (unsigned __int8 *)":", 1u);
      }
      else
      {
        *v18 = 58;
        ++v17[4];
      }
    }
    v6 = *(_BYTE *)(v7 - 16);
    v9 = (v6 & 2) != 0;
  }
  if ( v9 )
    v10 = *(_QWORD **)(v7 - 32);
  else
    v10 = (_QWORD *)(v8 - 8LL * ((v6 >> 2) & 0xF));
LABEL_20:
  v19 = v10[1];
  if ( v19 )
  {
    v20 = sub_B91420(v19);
    v22 = dest;
    v23 = (void *)v20;
    v24 = v90;
    v25 = v21;
    if ( v90 - (unsigned __int64)dest >= v21 )
    {
      if ( v21 )
      {
        memcpy(dest, v23, v21);
        v24 = v90;
        v22 = (char *)dest + v25;
        dest = (char *)dest + v25;
      }
      goto LABEL_24;
    }
    sub_CB6200((__int64)v89, (unsigned __int8 *)v23, v21);
  }
  v22 = dest;
  v24 = v90;
LABEL_24:
  if ( v24 - (unsigned __int64)v22 <= 3 )
  {
    sub_CB6200((__int64)v89, " <- ", 4u);
  }
  else
  {
    *v22 = 539835424;
    dest = (char *)dest + 4;
  }
  v26 = (unsigned __int64 *)sub_2E891C0(v2);
  v27 = (unsigned __int64 *)sub_B0D520((__int64)v26);
  v85 = v28;
  v84 = v27;
  if ( (_BYTE)v28 )
    v26 = v84;
  if ( (unsigned int)((__int64)(v26[3] - v26[2]) >> 3) )
  {
    v62 = dest;
    if ( (unsigned __int64)dest >= v90 )
    {
      sub_CB5D20((__int64)v89, 91);
    }
    else
    {
      dest = (char *)dest + 1;
      *v62 = 91;
    }
    v63 = v89;
    v82 = (unsigned __int64 *)v26[3];
    v84 = (unsigned __int64 *)v26[2];
    if ( v82 != v84 )
    {
      while ( 1 )
      {
        v64 = (unsigned __int8 *)sub_E06E20(*v84);
        v66 = (void *)v63[4];
        if ( v63[3] - (_QWORD)v66 < v65 )
        {
          sub_CB6200((__int64)v63, v64, v65);
        }
        else if ( v65 )
        {
          v78 = v65;
          memcpy(v66, v64, v65);
          v63[4] += v78;
        }
        LODWORD(v67) = 0;
        while ( (unsigned int)v67 < (unsigned int)sub_AF4160(&v84) - 1 )
        {
          v69 = dest;
          if ( (unsigned __int64)dest < v90 )
          {
            v68 = v89;
            dest = (char *)dest + 1;
            *v69 = 32;
          }
          else
          {
            v68 = (_QWORD *)sub_CB5D20((__int64)v89, 32);
          }
          v67 = (unsigned int)(v67 + 1);
          sub_CB59D0((__int64)v68, v84[v67]);
        }
        v70 = v84;
        v84 = &v70[(unsigned int)sub_AF4160(&v84)];
        if ( v82 == v84 )
          break;
        if ( v90 - (unsigned __int64)dest > 1 )
        {
          v63 = v89;
          *(_WORD *)dest = 8236;
          dest = (char *)dest + 2;
        }
        else
        {
          v63 = (_QWORD *)sub_CB6200((__int64)v89, (unsigned __int8 *)", ", 2u);
        }
      }
    }
    if ( v90 - (unsigned __int64)dest <= 1 )
    {
      sub_CB6200((__int64)v89, (unsigned __int8 *)"] ", 2u);
    }
    else
    {
      *(_WORD *)dest = 8285;
      dest = (char *)dest + 2;
    }
  }
  v29 = *(_QWORD *)(v2 + 32);
  v30 = *(_WORD *)(v2 + 68);
  v31 = v29;
  if ( v30 == 14 )
  {
    v81 = v29 + 40;
  }
  else
  {
    v31 = v29 + 80;
    v81 = v29 + 40LL * (*(_DWORD *)(v2 + 40) & 0xFFFFFF);
  }
  if ( v81 != v31 )
  {
    while ( 1 )
    {
      if ( v30 != 14 )
        v29 += 80;
      if ( v31 != v29 )
      {
        if ( v90 - (unsigned __int64)dest <= 1 )
        {
          sub_CB6200((__int64)v89, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *(_WORD *)dest = 8236;
          dest = (char *)dest + 2;
        }
      }
      switch ( *(_BYTE *)v31 )
      {
        case 0:
        case 5:
          v83 = 0;
          v32 = *(_BYTE *)v31;
          if ( *(_BYTE *)v31 )
          {
            v46 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a2 + 232) + 16LL) + 136LL);
            if ( v46 == sub_2DD19D0 )
LABEL_128:
              BUG();
            v47 = v46();
            v72 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, unsigned int *))(*(_QWORD *)v47 + 224LL))(
                    v47,
                    *(_QWORD *)(a2 + 232),
                    *(unsigned int *)(v31 + 24),
                    &v83);
            v32 = 1;
            if ( !v83 )
            {
LABEL_67:
              v48 = dest;
              if ( v90 - (unsigned __int64)dest <= 4 )
              {
                sub_CB6200((__int64)v89, (unsigned __int8 *)"undef", 5u);
              }
              else
              {
                *(_DWORD *)dest = 1701080693;
                v48[4] = 102;
                dest = (char *)dest + 5;
              }
              goto LABEL_46;
            }
          }
          else
          {
            v72 = 0;
            v83 = *(_DWORD *)(v31 + 8);
            if ( !v83 )
              goto LABEL_67;
          }
          v76 = v32;
          if ( *(_WORD *)(v2 + 68) == 14 && (v56 = *(_QWORD *)(v2 + 32), *(_BYTE *)(v56 + 40) == 1) && !*(_BYTE *)v56 )
          {
            v72 = *(_QWORD *)(v56 + 64);
          }
          else if ( !v32 )
          {
            goto LABEL_42;
          }
          v57 = dest;
          if ( (unsigned __int64)dest >= v90 )
          {
            sub_CB5D20((__int64)v89, 91);
            v76 = 1;
          }
          else
          {
            v76 = 1;
            dest = (char *)dest + 1;
            *v57 = 91;
          }
LABEL_42:
          v33 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a2 + 232) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a2 + 232) + 16LL));
          v34 = v83;
          sub_2FF6320((__int64 *)&v84, v83, v33, 0, 0);
          if ( !v86 )
            sub_4263D6(&v84, v34, v35);
          v87(&v84, v89);
          if ( v86 )
            v86(&v84, &v84, 3);
          if ( v76 )
          {
            v58 = dest;
            if ( (unsigned __int64)dest >= v90 )
            {
              v59 = (_QWORD *)sub_CB5D20((__int64)v89, 43);
            }
            else
            {
              v59 = v89;
              dest = (char *)dest + 1;
              *v58 = 43;
            }
            v60 = sub_CB59F0((__int64)v59, v72);
            v61 = *(_BYTE **)(v60 + 32);
            if ( (unsigned __int64)v61 >= *(_QWORD *)(v60 + 24) )
            {
              sub_CB5D20(v60, 93);
            }
            else
            {
              *(_QWORD *)(v60 + 32) = v61 + 1;
              *v61 = 93;
            }
          }
          goto LABEL_46;
        case 1:
          sub_CB59F0((__int64)v89, *(_QWORD *)(v31 + 24));
          goto LABEL_46;
        case 2:
          sub_C49420(*(_QWORD *)(v31 + 24) + 24LL, (__int64)v89, 0);
          goto LABEL_46;
        case 3:
          v73 = *(_QWORD *)(v31 + 24);
          v77 = (unsigned __int64 *)sub_C33340();
          v44 = (__int64 *)(v73 + 24);
          if ( *(unsigned __int64 **)(v73 + 24) == v77 )
          {
            sub_C3C790(&v84, (_QWORD **)v44);
            if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v31 + 24) + 8LL) + 8LL) <= 3u )
            {
LABEL_60:
              v45 = sub_C41B00((__int64 *)&v84);
              sub_CB5AB0((__int64)v89, v45);
              goto LABEL_61;
            }
          }
          else
          {
            sub_C33EB0(&v84, v44);
            if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v31 + 24) + 8LL) + 8LL) <= 3u )
              goto LABEL_60;
          }
          v52 = sub_C33320();
          sub_C41640((__int64 *)&v84, v52, 1, (bool *)&v83);
          v53 = dest;
          if ( v90 - (unsigned __int64)dest <= 0xD )
          {
            v54 = (_QWORD *)sub_CB6200((__int64)v89, "(long double) ", 0xEu);
          }
          else
          {
            *((_DWORD *)dest + 2) = 1701601909;
            v54 = v89;
            *(_QWORD *)v53 = 0x6F6420676E6F6C28LL;
            v53[6] = 8233;
            dest = (char *)dest + 14;
          }
          v75 = (__int64)v54;
          v55 = sub_C41B00((__int64 *)&v84);
          sub_CB5AB0(v75, v55);
LABEL_61:
          if ( v77 == v84 )
          {
            if ( v85 )
            {
              v49 = *(v85 - 1);
              v50 = &v85[3 * v49];
              if ( v85 != v50 )
              {
                v74 = v2;
                v51 = &v85[3 * v49];
                do
                {
                  v51 -= 3;
                  sub_91D830(v51);
                }
                while ( v85 != v51 );
                v50 = v51;
                v2 = v74;
              }
              j_j_j___libc_free_0_0((unsigned __int64)(v50 - 1));
            }
          }
          else
          {
            sub_C338F0((__int64)&v84);
          }
          goto LABEL_46;
        case 7:
          v36 = dest;
          if ( v90 - (unsigned __int64)dest <= 0xD )
          {
            v37 = (_QWORD *)sub_CB6200((__int64)v89, "!target-index(", 0xEu);
          }
          else
          {
            *((_DWORD *)dest + 2) = 1701080681;
            v37 = v89;
            *(_QWORD *)v36 = 0x2D74656772617421LL;
            v36[6] = 10360;
            dest = (char *)dest + 14;
          }
          v38 = sub_CB59F0((__int64)v37, *(int *)(v31 + 24));
          v39 = *(_BYTE **)(v38 + 32);
          if ( *(_BYTE **)(v38 + 24) == v39 )
          {
            v38 = sub_CB6200(v38, (unsigned __int8 *)",", 1u);
          }
          else
          {
            *v39 = 44;
            ++*(_QWORD *)(v38 + 32);
          }
          v40 = sub_CB59F0(v38, *(unsigned int *)(v31 + 8) | (unsigned __int64)((__int64)*(int *)(v31 + 32) << 32));
          v41 = *(_BYTE **)(v40 + 32);
          if ( *(_BYTE **)(v40 + 24) == v41 )
          {
            sub_CB6200(v40, (unsigned __int8 *)")", 1u);
LABEL_46:
            v31 += 40;
            if ( v81 == v31 )
              goto LABEL_54;
          }
          else
          {
            *v41 = 41;
            v31 += 40;
            ++*(_QWORD *)(v40 + 32);
            if ( v81 == v31 )
              goto LABEL_54;
          }
          v30 = *(_WORD *)(v2 + 68);
          v29 = *(_QWORD *)(v2 + 32);
          break;
        default:
          goto LABEL_128;
      }
    }
  }
LABEL_54:
  v42 = *(_QWORD *)(a2 + 224);
  v43 = *(void (__fastcall **)(__int64, unsigned __int64 **, __int64))(*(_QWORD *)v42 + 136LL);
  v88 = 261;
  v84 = v94;
  v85 = v95;
  v43(v42, &v84, 1);
  v89[0] = &unk_49DD388;
  sub_CB5840((__int64)v89);
  if ( v94 != (unsigned __int64 *)v97 )
    _libc_free((unsigned __int64)v94);
  return 1;
}
