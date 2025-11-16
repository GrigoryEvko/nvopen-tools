// Function: sub_98DB10
// Address: 0x98db10
//
__int64 __fastcall sub_98DB10(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // edx
  unsigned int v12; // ebx
  int v13; // eax
  __int64 v14; // r12
  __int64 v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rbx
  __int64 *v26; // rbx
  __int64 *v27; // r15
  unsigned __int8 *v28; // rax
  __int64 *v29; // rax
  __int64 *v30; // rdx
  unsigned __int8 *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  char v34; // di
  unsigned __int8 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r9
  unsigned __int8 *v38; // rax
  char v39; // dl
  char v40; // r8
  int v41; // [rsp+Ch] [rbp-124h]
  int v42; // [rsp+Ch] [rbp-124h]
  __int64 v43; // [rsp+10h] [rbp-120h]
  int v44; // [rsp+10h] [rbp-120h]
  __int64 v45; // [rsp+18h] [rbp-118h]
  unsigned __int8 v46; // [rsp+18h] [rbp-118h]
  unsigned __int8 v47; // [rsp+18h] [rbp-118h]
  __int64 v48; // [rsp+20h] [rbp-110h] BYREF
  __int64 *v49; // [rsp+28h] [rbp-108h]
  __int64 v50; // [rsp+30h] [rbp-100h]
  int v51; // [rsp+38h] [rbp-F8h]
  unsigned __int8 v52; // [rsp+3Ch] [rbp-F4h]
  __int64 v53; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v54; // [rsp+60h] [rbp-D0h] BYREF
  unsigned __int8 *v55; // [rsp+68h] [rbp-C8h]
  __int64 v56; // [rsp+70h] [rbp-C0h]
  int v57; // [rsp+78h] [rbp-B8h]
  char v58; // [rsp+7Ch] [rbp-B4h]
  __int64 v59; // [rsp+80h] [rbp-B0h] BYREF

  if ( *(_BYTE *)a1 <= 0x1Cu )
  {
    if ( *(_BYTE *)a1 != 22 || (unsigned __int8)sub_B2FC80(*(_QWORD *)(a1 + 24)) )
      return 0;
    v23 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 80LL);
    if ( !v23 )
      BUG();
    v5 = *(_QWORD *)(v23 + 32);
    v43 = v23 - 24;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 32);
    v43 = *(_QWORD *)(a1 + 40);
  }
  v45 = v43 + 48;
  if ( (_BYTE)a2 )
  {
    v57 = 0;
    v55 = (unsigned __int8 *)&v59;
    v49 = &v53;
    v56 = 0x100000010LL;
    v50 = 0x100000004LL;
    v58 = 1;
    v54 = 1;
    v51 = 0;
    v52 = 1;
    v53 = v43;
    v48 = 1;
    v42 = 32;
    v59 = a1;
    while ( 1 )
    {
      if ( v5 == v45 )
        goto LABEL_86;
      do
      {
        if ( !v5 )
          BUG();
        if ( *(_BYTE *)(v5 - 24) == 85 )
        {
          v32 = *(_QWORD *)(v5 - 56);
          if ( v32 )
          {
            if ( !*(_BYTE *)v32 )
            {
              a4 = *(_QWORD *)(v5 + 56);
              if ( *(_QWORD *)(v32 + 24) == a4
                && (*(_BYTE *)(v32 + 33) & 0x20) != 0
                && (unsigned int)(*(_DWORD *)(v32 + 36) - 68) <= 3 )
              {
                goto LABEL_85;
              }
            }
          }
        }
        if ( !--v42 )
          goto LABEL_140;
        v24 = v5 - 24;
        a2 = (__int64)&v54;
        result = sub_98D5C0((unsigned __int8 *)(v5 - 24), (__int64)&v54, a3, a4, a5);
        if ( (_BYTE)result || (result = sub_98CD80((char *)(v5 - 24)), !(_BYTE)result) )
        {
          LOBYTE(v30) = v52;
          goto LABEL_136;
        }
        v25 = 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
        {
          a4 = *(_QWORD *)(v5 - 32);
          v26 = (__int64 *)(a4 + v25);
        }
        else
        {
          a4 = v24 - v25;
          v26 = (__int64 *)(v5 - 24);
        }
        v27 = (__int64 *)a4;
        if ( (__int64 *)a4 == v26 )
          goto LABEL_84;
        while ( 1 )
        {
          a2 = *v27;
          if ( v58 )
            break;
          if ( sub_C8CA60(&v54, a2, a3, a4, a5) && (unsigned __int8)sub_98D0D0((__int64)v27, a2, (__int64)a3, a4, a5) )
            goto LABEL_95;
LABEL_83:
          v27 += 4;
          if ( v26 == v27 )
            goto LABEL_84;
        }
        v28 = v55;
        a3 = &v55[8 * HIDWORD(v56)];
        if ( v55 == a3 )
          goto LABEL_83;
        while ( a2 != *(_QWORD *)v28 )
        {
          v28 += 8;
          if ( a3 == v28 )
            goto LABEL_83;
        }
        if ( !(unsigned __int8)sub_98D0D0((__int64)v27, a2, (__int64)a3, a4, a5) )
          goto LABEL_83;
LABEL_95:
        if ( !v58 )
          goto LABEL_126;
        v31 = v55;
        a4 = HIDWORD(v56);
        a3 = &v55[8 * HIDWORD(v56)];
        if ( v55 == a3 )
        {
LABEL_125:
          if ( HIDWORD(v56) < (unsigned int)v56 )
          {
            a4 = (unsigned int)++HIDWORD(v56);
            *(_QWORD *)a3 = v24;
            ++v54;
          }
          else
          {
LABEL_126:
            a2 = v5 - 24;
            sub_C8CC70(&v54, v5 - 24);
          }
        }
        else
        {
          while ( v24 != *(_QWORD *)v31 )
          {
            v31 += 8;
            if ( a3 == v31 )
              goto LABEL_125;
          }
        }
LABEL_84:
        if ( *(_BYTE *)(v5 - 24) == 86 )
        {
          a4 = *(_BYTE *)(v5 - 17) & 0x40;
          if ( (*(_BYTE *)(v5 - 17) & 0x40) != 0 )
          {
            v33 = *(_QWORD *)(v5 - 32);
            v34 = v58;
            a2 = *(_QWORD *)(v33 + 32);
            if ( v58 )
              goto LABEL_110;
LABEL_128:
            if ( !sub_C8CA60(&v54, a2, v33, a4, a5) )
              goto LABEL_85;
            v34 = v58;
            a4 = *(_BYTE *)(v5 - 17) & 0x40;
          }
          else
          {
            v34 = v58;
            v33 = v24 - 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF);
            a2 = *(_QWORD *)(v33 + 32);
            if ( !v58 )
              goto LABEL_128;
LABEL_110:
            v35 = v55;
            a3 = &v55[8 * HIDWORD(v56)];
            if ( v55 == a3 )
              goto LABEL_85;
            while ( a2 != *(_QWORD *)v35 )
            {
              v35 += 8;
              if ( a3 == v35 )
                goto LABEL_85;
            }
          }
          if ( (_BYTE)a4 )
            v36 = *(_QWORD *)(v5 - 32);
          else
            v36 = v24 - 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF);
          v37 = *(_QWORD *)(v36 + 64);
          if ( v34 )
          {
            a3 = v55;
            a4 = (__int64)&v55[8 * HIDWORD(v56)];
            a2 = HIDWORD(v56);
            if ( v55 != (unsigned __int8 *)a4 )
            {
              v38 = v55;
              do
              {
                if ( v37 == *(_QWORD *)v38 )
                  goto LABEL_121;
                v38 += 8;
              }
              while ( (unsigned __int8 *)a4 != v38 );
            }
            goto LABEL_85;
          }
          a2 = *(_QWORD *)(v36 + 64);
          if ( !sub_C8CA60(&v54, a2, v36, a4, a5) )
            goto LABEL_85;
          if ( v58 )
          {
            a3 = v55;
            a4 = (__int64)&v55[8 * HIDWORD(v56)];
            a2 = HIDWORD(v56);
            if ( v55 != (unsigned __int8 *)a4 )
            {
LABEL_121:
              while ( v24 != *(_QWORD *)a3 )
              {
                a3 += 8;
                if ( (unsigned __int8 *)a4 == a3 )
                  goto LABEL_123;
              }
              goto LABEL_85;
            }
LABEL_123:
            if ( (unsigned int)v56 > (unsigned int)a2 )
            {
              a2 = (unsigned int)(a2 + 1);
              HIDWORD(v56) = a2;
              *(_QWORD *)a4 = v24;
              ++v54;
              goto LABEL_85;
            }
          }
          a2 = v5 - 24;
          sub_C8CC70(&v54, v5 - 24);
        }
LABEL_85:
        v5 = *(_QWORD *)(v5 + 8);
      }
      while ( v5 != v45 );
LABEL_86:
      v43 = sub_AA56F0(v43);
      if ( !v43 )
      {
LABEL_140:
        LOBYTE(v30) = v52;
        goto LABEL_141;
      }
      if ( !v52 )
        goto LABEL_133;
      v29 = v49;
      v30 = &v49[HIDWORD(v50)];
      if ( v49 != v30 )
      {
        while ( v43 != *v29 )
        {
          if ( v30 == ++v29 )
            goto LABEL_132;
        }
        result = 0;
        goto LABEL_138;
      }
LABEL_132:
      if ( HIDWORD(v50) >= (unsigned int)v50 )
      {
LABEL_133:
        a2 = v43;
        sub_C8CC70(&v48, v43);
        v40 = v39;
        v30 = (__int64 *)v52;
        if ( v40 )
          goto LABEL_134;
LABEL_141:
        result = 0;
LABEL_136:
        if ( !(_BYTE)v30 )
        {
          v46 = result;
          _libc_free(v49, a2);
          result = v46;
        }
LABEL_138:
        if ( !v58 )
        {
          v47 = result;
          _libc_free(v55, a2);
          return v47;
        }
        return result;
      }
      ++HIDWORD(v50);
      *v30 = v43;
      ++v48;
LABEL_134:
      v5 = sub_AA4FF0(v43, a2, v30);
      v45 = v43 + 48;
    }
  }
  if ( v43 + 48 == v5 )
    return 0;
  v41 = 32;
  while ( 1 )
  {
    if ( !v5 )
      BUG();
    v6 = v5 - 24;
    if ( *(_BYTE *)(v5 - 24) != 85 )
      break;
    v10 = *(_QWORD *)(v5 - 56);
    if ( v10 && !*(_BYTE *)v10 && *(_QWORD *)(v10 + 24) == *(_QWORD *)(v5 + 56) && (*(_BYTE *)(v10 + 33) & 0x20) != 0 )
    {
      if ( (unsigned int)(*(_DWORD *)(v10 + 36) - 68) <= 3 )
        goto LABEL_13;
      if ( !--v41 )
        return 0;
    }
    else if ( !--v41 )
    {
      return 0;
    }
LABEL_23:
    if ( (unsigned __int8)sub_B491E0(v5 - 24) && a1 == *(_QWORD *)(v5 - 56) )
      return 1;
    v11 = *(unsigned __int8 *)(v5 - 24);
    v12 = 0;
    v13 = v11 - 29;
    if ( v11 != 40 )
    {
LABEL_26:
      v14 = 0;
      if ( v13 != 56 )
      {
        if ( v13 != 5 )
          BUG();
        v14 = 64;
      }
      if ( *(char *)(v5 - 17) >= 0 )
        goto LABEL_57;
LABEL_47:
      v17 = sub_BD2BC0(v5 - 24);
      if ( *(char *)(v5 - 17) >= 0 )
      {
        if ( !(unsigned int)((v17 + v18) >> 4) )
          goto LABEL_57;
      }
      else
      {
        if ( !(unsigned int)((v17 + v18 - sub_BD2BC0(v5 - 24)) >> 4) )
          goto LABEL_57;
        if ( *(char *)(v5 - 17) < 0 )
        {
          v44 = *(_DWORD *)(sub_BD2BC0(v5 - 24) + 8);
          if ( *(char *)(v5 - 17) >= 0 )
            BUG();
          v19 = sub_BD2BC0(v5 - 24);
          v21 = 32LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v44);
          goto LABEL_52;
        }
      }
      BUG();
    }
    while ( 1 )
    {
      v14 = 32LL * (unsigned int)sub_B491D0(v5 - 24);
      if ( *(char *)(v5 - 17) < 0 )
        goto LABEL_47;
LABEL_57:
      v21 = 0;
LABEL_52:
      if ( v12 >= (unsigned int)((32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) - 32 - v14 - v21) >> 5) )
        break;
      if ( ((unsigned __int8)sub_B49B80(v5 - 24, v12, 40)
         || (unsigned __int8)sub_B49B80(v5 - 24, v12, 90)
         || (unsigned __int8)sub_B49B80(v5 - 24, v12, 91))
        && a1 == *(_QWORD *)(v6 + 32 * (v12 - (unsigned __int64)(*(_DWORD *)(v5 - 20) & 0x7FFFFFF))) )
      {
        return 1;
      }
      v22 = *(unsigned __int8 *)(v5 - 24);
      ++v12;
      v13 = v22 - 29;
      if ( v22 != 40 )
        goto LABEL_26;
    }
LABEL_12:
    if ( !(unsigned __int8)sub_98CD80((char *)(v5 - 24)) )
      return 0;
LABEL_13:
    v5 = *(_QWORD *)(v5 + 8);
    if ( v45 == v5 )
      return 0;
  }
  if ( --v41 )
  {
    switch ( *(_BYTE *)(v5 - 24) )
    {
      case 0x1E:
        v15 = sub_B43CB0(v5 - 24);
        if ( (unsigned __int8)sub_B2D630(v15, 40) )
        {
          v16 = (*(_BYTE *)(v5 - 17) & 0x40) != 0
              ? *(_QWORD **)(v5 - 32)
              : (_QWORD *)(v6 - 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF));
          if ( a1 == *v16 )
            return 1;
        }
        goto LABEL_12;
      case 0x1F:
        if ( (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) == 3 && a1 == *(_QWORD *)(v5 - 120) )
          return 1;
        goto LABEL_12;
      case 0x20:
        v9 = **(_QWORD **)(v5 - 32);
        if ( a1 == v9 )
          goto LABEL_17;
        goto LABEL_12;
      case 0x22:
        goto LABEL_23;
      case 0x3D:
      case 0x3E:
        v7 = *(_QWORD *)(v5 - 56);
        if ( v7 )
          goto LABEL_11;
        goto LABEL_12;
      case 0x41:
        v9 = *(_QWORD *)(v5 - 120);
        if ( a1 != v9 )
          goto LABEL_12;
LABEL_17:
        if ( v9 )
          return 1;
        goto LABEL_12;
      case 0x42:
        v7 = *(_QWORD *)(v5 - 88);
        if ( !v7 )
          goto LABEL_12;
LABEL_11:
        if ( a1 != v7 )
          goto LABEL_12;
        return 1;
      default:
        goto LABEL_12;
    }
  }
  return 0;
}
