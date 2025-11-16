// Function: sub_32507E0
// Address: 0x32507e0
//
__int64 __fastcall sub_32507E0(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  bool v7; // dl
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // r14d
  size_t v12; // rdx
  __int64 v13; // rdx
  __int64 result; // rax
  int v15; // eax
  int v16; // edx
  __int64 v17; // r8
  __int64 v18; // r8
  const void *v19; // rcx
  size_t v20; // rdx
  size_t v21; // r8
  bool v22; // dl
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int8 v25; // dl
  __int64 v26; // r9
  __int64 v27; // rax
  unsigned __int8 **v28; // rbx
  unsigned __int8 *v29; // r14
  unsigned __int8 v30; // al
  int v31; // eax
  unsigned __int8 v32; // al
  __int64 v33; // rdx
  unsigned __int64 v34; // rax
  int v35; // eax
  __int64 v36; // r8
  unsigned __int8 v37; // al
  __int64 v38; // rdx
  __int64 v39; // rsi
  unsigned __int64 v40; // rax
  __int16 v41; // ax
  __int64 v42; // r15
  unsigned __int8 v43; // al
  __int64 *v44; // rdx
  const void *v45; // rcx
  size_t v46; // rdx
  size_t v47; // r8
  unsigned __int8 v48; // al
  unsigned __int8 *v49; // rdx
  __int64 v50; // rdx
  unsigned __int8 v51; // al
  __int64 v52; // rdi
  const void *v53; // rax
  size_t v54; // rdx
  unsigned __int8 *v55; // rdx
  __int64 v56; // rdi
  const void *v57; // rax
  size_t v58; // rdx
  __int64 v59; // r8
  __int64 v60; // rdx
  __int64 v61; // rsi
  unsigned __int64 v62; // rax
  __int16 v63; // ax
  __int64 v64; // rax
  unsigned __int8 *v65; // r15
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rsi
  unsigned __int8 v68; // al
  __int64 v69; // r9
  __int64 v70; // r15
  _BYTE *v71; // rax
  _BYTE *v72; // rdx
  unsigned __int8 v73; // al
  __int64 v74; // rcx
  char v75; // al
  unsigned __int8 *v76; // [rsp+10h] [rbp-90h]
  _BYTE *v77; // [rsp+10h] [rbp-90h]
  __int64 v78; // [rsp+18h] [rbp-88h]
  __int64 v79; // [rsp+20h] [rbp-80h]
  unsigned __int8 **v80; // [rsp+28h] [rbp-78h]
  size_t v81; // [rsp+30h] [rbp-70h]
  const void *v82; // [rsp+38h] [rbp-68h]
  size_t v83; // [rsp+40h] [rbp-60h]
  __int64 v84; // [rsp+48h] [rbp-58h]
  unsigned __int16 v85; // [rsp+54h] [rbp-4Ch]
  bool v86; // [rsp+57h] [rbp-49h]
  __int64 v87; // [rsp+58h] [rbp-48h]
  int v88; // [rsp+6Ch] [rbp-34h]

  v5 = a3;
  v87 = a3 - 16;
  v6 = *(_BYTE *)(a3 - 16);
  v7 = (v6 & 2) != 0;
  if ( (v6 & 2) != 0 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 16LL);
    if ( v8 )
      goto LABEL_3;
  }
  else
  {
    v8 = *(_QWORD *)(v87 - 8LL * ((v6 >> 2) & 0xF) + 16);
    if ( v8 )
    {
LABEL_3:
      v9 = sub_B91420(v8);
      v11 = *(unsigned __int16 *)(a2 + 28);
      v82 = (const void *)v9;
      v83 = v12;
      v81 = v12;
      v84 = *(_QWORD *)(v5 + 24) >> 3;
      switch ( (__int16)v11 )
      {
        case 1:
          sub_324E550(a1, a2, v5);
          v86 = 0;
          break;
        case 2:
        case 19:
        case 23:
        case 43:
        case 51:
          v6 = *(_BYTE *)(v5 - 16);
          goto LABEL_46;
        case 4:
          sub_324EE90(a1, a2, v5);
          v86 = 0;
          break;
        default:
          v86 = 0;
          break;
      }
      goto LABEL_5;
    }
  }
  v11 = *(unsigned __int16 *)(a2 + 28);
  v82 = 0;
  v81 = 0;
  v10 = *(_QWORD *)(v5 + 24) >> 3;
  v84 = v10;
  switch ( (__int16)v11 )
  {
    case 1:
      sub_324E550(a1, a2, v5);
      v86 = 0;
      goto LABEL_7;
    case 2:
    case 19:
    case 23:
    case 43:
    case 51:
      v83 = 0;
LABEL_46:
      v86 = (_WORD)v11 == 2;
      if ( (_WORD)v11 == 51 )
      {
        v22 = (v6 & 2) != 0;
        if ( (v6 & 2) != 0 )
          v61 = *(_QWORD *)(v5 - 32);
        else
          v61 = v87 - 8LL * ((v6 >> 2) & 0xF);
        v10 = *(_QWORD *)(v61 + 64);
        v79 = v10;
        if ( v10 )
        {
          v62 = sub_324F1B0(a1, a2, v10);
          sub_32494F0(a1, a2, 21, v62);
          v6 = *(_BYTE *)(v5 - 16);
          v22 = (v6 & 2) != 0;
        }
      }
      else
      {
        if ( (v11 & 0xFFFB) == 0x13 || (_WORD)v11 == 2 )
        {
          if ( (v6 & 2) != 0 )
            v60 = *(_QWORD *)(v5 - 32);
          else
            v60 = v87 - 8LL * ((v6 >> 2) & 0xF);
          sub_324D230(a1, a2, *(_QWORD *)(v60 + 48));
          v6 = *(_BYTE *)(v5 - 16);
        }
        v79 = 0;
        v22 = (v6 & 2) != 0;
      }
      if ( v22 )
        v23 = *(_QWORD *)(v5 - 32);
      else
        v23 = v87 - 8LL * ((v6 >> 2) & 0xF);
      v24 = *(_QWORD *)(v23 + 32);
      if ( !v24 )
        goto LABEL_62;
      v25 = *(_BYTE *)(v24 - 16);
      if ( (v25 & 2) != 0 )
      {
        v26 = *(_QWORD *)(v24 - 32);
        v27 = *(unsigned int *)(v24 - 24);
      }
      else
      {
        v26 = v24 - 16 - 8LL * ((v25 >> 2) & 0xF);
        v27 = (*(_WORD *)(v24 - 16) >> 6) & 0xF;
      }
      v80 = (unsigned __int8 **)(v26 + 8 * v27);
      if ( v80 == (unsigned __int8 **)v26 )
        goto LABEL_62;
      v85 = v11;
      v78 = v5;
      v28 = (unsigned __int8 **)v26;
      break;
    case 4:
      sub_324EE90(a1, a2, v5);
      v86 = 0;
      goto LABEL_7;
    default:
      v86 = 0;
      if ( *(_DWORD *)(v5 + 44) == 30 )
        goto LABEL_33;
      goto LABEL_8;
  }
  do
  {
    v29 = *v28;
    if ( !*v28 )
      goto LABEL_60;
    v30 = *v29;
    if ( *v29 == 18 )
    {
      sub_3250680(a1, *v28, 0);
      goto LABEL_60;
    }
    switch ( v30 )
    {
      case 0xDu:
        if ( (unsigned __int16)sub_AF18C0((__int64)*v28) == 42 )
        {
          v67 = sub_324C6D0(a1, 42, a2, 0);
          v68 = *(v29 - 16);
          if ( (v68 & 2) != 0 )
            v69 = *((_QWORD *)v29 - 4);
          else
            v69 = (__int64)&v29[-8 * ((v68 >> 2) & 0xF) - 16];
          sub_32495E0(a1, v67, *(_QWORD *)(v69 + 24), 65);
        }
        else if ( (v29[21] & 0x10) != 0 )
        {
          sub_324F7C0(a1, (__int64)v29);
        }
        else if ( v85 == 51 )
        {
          v70 = sub_324C6D0(a1, 25, a2, 0);
          v71 = (_BYTE *)sub_AF2E00((__int64)v29);
          v72 = v71;
          if ( v71 && *v71 == 17 )
          {
            v73 = *(_BYTE *)(v79 - 16);
            if ( (v73 & 2) != 0 )
              v74 = *(_QWORD *)(v79 - 32);
            else
              v74 = v79 - 16 - 8LL * ((v73 >> 2) & 0xF);
            v77 = v72;
            v75 = sub_32120E0(*(char **)(v74 + 24));
            sub_324A260(a1, v70, 22, (__int64)(v77 + 24), v75);
          }
          sub_324F1B0(a1, v70, (__int64)v29);
        }
        else
        {
          sub_324F1B0(a1, a2, (__int64)v29);
        }
        break;
      case 0x1Cu:
        v41 = sub_AF18C0((__int64)*v28);
        v42 = sub_324C6D0(a1, v41, a2, 0);
        v76 = v29 - 16;
        v43 = *(v29 - 16);
        if ( (v43 & 2) != 0 )
          v44 = (__int64 *)*((_QWORD *)v29 - 4);
        else
          v44 = (__int64 *)&v76[-8 * ((v43 >> 2) & 0xF)];
        v45 = (const void *)*v44;
        if ( *v44 )
        {
          v45 = (const void *)sub_B91420(*v44);
          v47 = v46;
        }
        else
        {
          v47 = 0;
        }
        sub_324AD70(a1, v42, 16360, v45, v47);
        v48 = *(v29 - 16);
        if ( (v48 & 2) != 0 )
          v49 = (unsigned __int8 *)*((_QWORD *)v29 - 4);
        else
          v49 = &v76[-8 * ((v48 >> 2) & 0xF)];
        v50 = *((_QWORD *)v49 + 4);
        if ( v50 )
          sub_32495E0(a1, v42, v50, 73);
        sub_3249E50(a1, v42, (__int64)v29);
        v51 = *(v29 - 16);
        if ( (v51 & 2) != 0 )
        {
          v52 = *(_QWORD *)(*((_QWORD *)v29 - 4) + 16LL);
          if ( !v52 )
          {
LABEL_100:
            v55 = (unsigned __int8 *)*((_QWORD *)v29 - 4);
            goto LABEL_101;
          }
        }
        else
        {
          v10 = (__int64)&v76[-8 * ((v51 >> 2) & 0xF)];
          v52 = *(_QWORD *)(v10 + 16);
          if ( !v52 )
            goto LABEL_133;
        }
        v53 = (const void *)sub_B91420(v52);
        if ( v54 )
          sub_324AD70(a1, v42, 16361, v53, v54);
        v51 = *(v29 - 16);
        if ( (v51 & 2) != 0 )
          goto LABEL_100;
LABEL_133:
        v55 = &v76[-8 * ((v51 >> 2) & 0xF)];
LABEL_101:
        v56 = *((_QWORD *)v55 + 3);
        if ( v56 )
        {
          v57 = (const void *)sub_B91420(v56);
          if ( v58 )
            sub_324AD70(a1, v42, 16362, v57, v58);
        }
        v59 = *((unsigned int *)v29 + 5);
        if ( (_DWORD)v59 )
        {
          BYTE2(v88) = 0;
          sub_3249A20(a1, (unsigned __int64 **)(v42 + 8), 16363, v88, v59);
        }
        break;
      case 0xEu:
        if ( (unsigned __int16)sub_AF18C0((__int64)*v28) == 51 )
        {
          v63 = sub_AF18C0((__int64)v29);
          v64 = sub_324C6D0(a1, v63, a2, 0);
          sub_32507E0(a1, v64, v29);
        }
        break;
      default:
        if ( v85 == 43 )
        {
          v65 = sub_3247C80((__int64)a1, v29);
          if ( v65 )
          {
            v66 = sub_324C6D0(a1, 44, a2, 0);
            sub_32494F0(a1, v66, 68, (unsigned __int64)v65);
          }
        }
        break;
    }
LABEL_60:
    ++v28;
  }
  while ( v80 != v28 );
  v11 = v85;
  v5 = v78;
LABEL_62:
  v31 = *(_DWORD *)(v5 + 20);
  if ( (v31 & 8) != 0 )
  {
    sub_3249FA0(a1, a2, 16356);
    v31 = *(_DWORD *)(v5 + 20);
  }
  if ( (v31 & 0x8000) != 0 )
    sub_3249FA0(a1, a2, 137);
  v32 = *(_BYTE *)(v5 - 16);
  if ( (v32 & 2) != 0 )
    v33 = *(_QWORD *)(v5 - 32);
  else
    v33 = v87 - 8LL * ((v32 >> 2) & 0xF);
  if ( *(_QWORD *)(v33 + 40) )
  {
    v34 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 40))(a1);
    sub_32494F0(a1, a2, 29, v34);
  }
  if ( (*(_BYTE *)(v5 + 21) & 2) != 0 )
    sub_3249FA0(a1, a2, 16364);
  if ( (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0 || (unsigned __int16)sub_3220AA0(a1[26]) > 4u )
  {
    v35 = *(_DWORD *)(v5 + 20);
    if ( (v35 & 0x400000) != 0 )
    {
      v36 = 5;
    }
    else
    {
      v36 = 4;
      if ( (v35 & 0x800000) == 0 )
        goto LABEL_76;
    }
    v88 = 65547;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 54, 65547, v36);
  }
LABEL_76:
  v37 = *(_BYTE *)(v5 - 16);
  if ( (v37 & 2) != 0 )
    v38 = *(_QWORD *)(v5 - 32);
  else
    v38 = v87 - 8LL * ((v37 >> 2) & 0xF);
  v39 = *(_QWORD *)(v38 + 112);
  if ( v39 )
  {
    v40 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(*a1 + 48))(a1, v39, v38, v10, v36);
    sub_32494F0(a1, a2, 71, v40);
  }
LABEL_5:
  if ( v83 )
    sub_324AD70(a1, a2, 3, v82, v81);
LABEL_7:
  v6 = *(_BYTE *)(v5 - 16);
  v7 = (v6 & 2) != 0;
  if ( *(_DWORD *)(v5 + 44) == 30 )
  {
LABEL_33:
    if ( v7 )
    {
      v13 = *(_QWORD *)(v5 - 32);
      if ( !*(_QWORD *)(v13 + 56) )
        goto LABEL_10;
    }
    else
    {
      v13 = v87 - 8LL * ((v6 >> 2) & 0xF);
      if ( !*(_QWORD *)(v13 + 56) )
        goto LABEL_10;
    }
    v19 = *(const void **)(v13 + 56);
    if ( v19 )
    {
      v19 = (const void *)sub_B91420(*(_QWORD *)(v13 + 56));
      v21 = v20;
    }
    else
    {
      v21 = 0;
    }
    sub_324AD70(a1, a2, 110, v19, v21);
    v6 = *(_BYTE *)(v5 - 16);
    v7 = (v6 & 2) != 0;
  }
LABEL_8:
  if ( v7 )
    v13 = *(_QWORD *)(v5 - 32);
  else
    v13 = v87 - 8LL * ((v6 >> 2) & 0xF);
LABEL_10:
  sub_324CC60(a1, a2, *(_QWORD *)(v13 + 104));
  if ( (_WORD)v11 == 4 || v86 || (result = v11 & 0xFFFFFFFB, (v11 & 0xFFFB) == 0x13) )
  {
    v15 = *(_DWORD *)(v5 + 20) & 4;
    if ( v84 )
    {
      if ( v15 && (_WORD)v11 != 4 )
      {
LABEL_16:
        sub_3249FA0(a1, a2, 60);
        v16 = *(_DWORD *)(v5 + 20);
        goto LABEL_17;
      }
      BYTE2(v88) = 0;
      sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 11, v88, v84);
    }
    else
    {
      if ( v15 )
        goto LABEL_16;
      BYTE2(v88) = 0;
      sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 11, v88, 0);
    }
    v16 = *(_DWORD *)(v5 + 20);
    if ( (v16 & 4) != 0 )
      goto LABEL_16;
LABEL_17:
    sub_3249F00(a1, a2, v16);
    if ( (*(_BYTE *)(v5 + 20) & 4) == 0 )
      sub_3249E10(a1, a2, v5);
    v17 = *(unsigned int *)(v5 + 44);
    if ( (_DWORD)v17 )
    {
      v88 = 65547;
      sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 16358, 65547, v17);
    }
    result = sub_3220AA0(a1[26]);
    if ( (unsigned __int16)result > 4u )
    {
      result = (unsigned int)sub_AF18D0(v5) >> 3;
      if ( (_DWORD)result )
      {
        v88 = 65551;
        result = sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 136, 65551, result & 0x1FFFFFFF);
      }
    }
    v18 = *(unsigned int *)(v5 + 40);
    if ( (_DWORD)v18 )
    {
      BYTE2(v88) = 0;
      return sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 15883, v88, v18);
    }
  }
  return result;
}
