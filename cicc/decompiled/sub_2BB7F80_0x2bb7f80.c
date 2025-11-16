// Function: sub_2BB7F80
// Address: 0x2bb7f80
//
void __fastcall sub_2BB7F80(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  unsigned __int64 *v7; // r14
  unsigned __int8 *v8; // r15
  unsigned __int8 *v9; // rbx
  char i; // al
  __int64 v11; // rcx
  unsigned __int8 **v12; // rdx
  unsigned __int8 **j; // rax
  bool v14; // r12
  unsigned __int8 **v15; // rax
  unsigned __int8 **v16; // rdx
  __int64 v17; // rdx
  _BYTE *v18; // rax
  unsigned __int8 *v19; // r12
  unsigned int v20; // ebx
  unsigned __int8 *v21; // r15
  char v22; // al
  unsigned __int8 **v23; // rdx
  __int64 v24; // rcx
  unsigned __int8 **k; // rax
  unsigned __int8 **v26; // rax
  unsigned __int8 **v27; // rdx
  unsigned __int64 *v28; // rbx
  unsigned int *v29; // r13
  __int64 v30; // r12
  __int64 v31; // rdi
  unsigned __int8 **v32; // rax
  unsigned __int8 **v33; // rax
  bool v34; // bl
  unsigned __int8 **v35; // rax
  __int64 v36; // rdi
  unsigned __int8 **v37; // rax
  unsigned __int8 **v38; // rax
  unsigned __int8 **v39; // rax
  unsigned int *v41; // [rsp+28h] [rbp-1C8h]
  unsigned __int64 v42[2]; // [rsp+40h] [rbp-1B0h] BYREF
  _BYTE v43[48]; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v44; // [rsp+80h] [rbp-170h] BYREF
  unsigned __int8 **v45; // [rsp+88h] [rbp-168h]
  __int64 v46; // [rsp+90h] [rbp-160h]
  int v47; // [rsp+98h] [rbp-158h]
  char v48; // [rsp+9Ch] [rbp-154h]
  _BYTE v49[128]; // [rsp+A0h] [rbp-150h] BYREF
  __int64 *v50; // [rsp+120h] [rbp-D0h] BYREF
  unsigned __int64 v51; // [rsp+128h] [rbp-C8h]
  __int64 v52; // [rsp+130h] [rbp-C0h] BYREF
  int v53; // [rsp+138h] [rbp-B8h]
  char v54; // [rsp+13Ch] [rbp-B4h]
  _BYTE v55[176]; // [rsp+140h] [rbp-B0h] BYREF

  if ( (unsigned int *)a1 == a2 || a2 == (unsigned int *)(a1 + 64) )
    return;
  v41 = (unsigned int *)(a1 + 64);
  do
  {
    v6 = 0;
    v7 = (unsigned __int64 *)v41;
    v8 = *(unsigned __int8 **)(*(_QWORD *)a1 + 8LL);
    v9 = *(unsigned __int8 **)(*(_QWORD *)v41 + 8LL);
    v44 = 0;
    v46 = 16;
    v45 = (unsigned __int8 **)v49;
    v47 = 0;
    v48 = 1;
    v50 = 0;
    v52 = 16;
    v53 = 0;
    v54 = 1;
    v51 = (unsigned __int64)v55;
    for ( i = 1; ; i = v48 )
    {
      if ( i )
      {
        v11 = (__int64)v45;
        v12 = &v45[HIDWORD(v46)];
        for ( j = v45; v12 != j; ++j )
        {
          if ( v8 == *j )
            goto LABEL_12;
        }
      }
      else if ( sub_C8CA60((__int64)&v44, (__int64)v8) )
      {
        goto LABEL_35;
      }
      if ( v54 )
        break;
      if ( sub_C8CA60((__int64)&v50, (__int64)v9) )
        goto LABEL_35;
LABEL_51:
      if ( v8 == v9 || (unsigned int)qword_500FC48 < v6 )
        goto LABEL_37;
      if ( !v48 )
      {
LABEL_65:
        sub_C8CC70((__int64)&v44, (__int64)v9, (__int64)v27, v11, a5, a6);
        goto LABEL_58;
      }
      v32 = v45;
      v11 = HIDWORD(v46);
      v27 = &v45[HIDWORD(v46)];
      if ( v45 == v27 )
      {
LABEL_68:
        if ( HIDWORD(v46) >= (unsigned int)v46 )
          goto LABEL_65;
        v11 = (unsigned int)++HIDWORD(v46);
        *v27 = v9;
        ++v44;
      }
      else
      {
        while ( v9 != *v32 )
        {
          if ( v27 == ++v32 )
            goto LABEL_68;
        }
      }
LABEL_58:
      if ( !v54 )
        goto LABEL_64;
      v33 = (unsigned __int8 **)v51;
      v11 = HIDWORD(v52);
      v27 = (unsigned __int8 **)(v51 + 8LL * HIDWORD(v52));
      if ( (unsigned __int8 **)v51 == v27 )
      {
LABEL_66:
        if ( HIDWORD(v52) >= (unsigned int)v52 )
        {
LABEL_64:
          sub_C8CC70((__int64)&v50, (__int64)v8, (__int64)v27, v11, a5, a6);
          goto LABEL_63;
        }
        ++HIDWORD(v52);
        *v27 = v8;
        v50 = (__int64 *)((char *)v50 + 1);
      }
      else
      {
        while ( v8 != *v33 )
        {
          if ( v27 == ++v33 )
            goto LABEL_66;
        }
      }
LABEL_63:
      ++v6;
      v9 = sub_98ACB0(v9, 1u);
      v8 = sub_98ACB0(v8, 1u);
    }
    v26 = (unsigned __int8 **)v51;
    v27 = (unsigned __int8 **)(v51 + 8LL * HIDWORD(v52));
    if ( (unsigned __int8 **)v51 == v27 )
      goto LABEL_51;
    while ( v9 != *v26 )
    {
      if ( v27 == ++v26 )
        goto LABEL_51;
    }
LABEL_35:
    if ( v48 )
    {
      v11 = (__int64)v45;
      v12 = &v45[HIDWORD(v46)];
      if ( v12 != v45 )
      {
LABEL_12:
        while ( v8 != *(unsigned __int8 **)v11 )
        {
          v11 += 8;
          if ( (unsigned __int8 **)v11 == v12 )
            goto LABEL_37;
        }
        goto LABEL_13;
      }
LABEL_37:
      v14 = 0;
    }
    else
    {
      if ( !sub_C8CA60((__int64)&v44, (__int64)v8) )
        goto LABEL_37;
LABEL_13:
      v14 = v54;
      if ( v54 )
      {
        v15 = (unsigned __int8 **)v51;
        v16 = (unsigned __int8 **)(v51 + 8LL * HIDWORD(v52));
        if ( (unsigned __int8 **)v51 == v16 )
          goto LABEL_40;
        while ( v9 != *v15 )
        {
          if ( v16 == ++v15 )
            goto LABEL_40;
        }
        v14 = 0;
        if ( !v48 )
          goto LABEL_19;
LABEL_41:
        v17 = v41[2];
        if ( v14 )
          goto LABEL_42;
        goto LABEL_20;
      }
      v14 = sub_C8CA60((__int64)&v50, (__int64)v9) == 0;
    }
    if ( !v54 )
      _libc_free(v51);
LABEL_40:
    if ( v48 )
      goto LABEL_41;
LABEL_19:
    _libc_free((unsigned __int64)v45);
    v17 = v41[2];
    if ( v14 )
    {
LABEL_42:
      v50 = &v52;
      v51 = 0x300000000LL;
      if ( (_DWORD)v17 )
        sub_2BB7BD0((__int64)&v50, (unsigned __int64 *)v41, v17, v11, a5, a6);
      v28 = (unsigned __int64 *)v41;
      v29 = v41 + 16;
      v30 = ((__int64)v41 - a1) >> 6;
      if ( (__int64)v41 - a1 > 0 )
      {
        do
        {
          v31 = (__int64)v28;
          v28 -= 8;
          sub_2BB7BD0(v31, v28, v17, v11, a5, a6);
          --v30;
        }
        while ( v30 );
      }
      sub_2BB7BD0(a1, (unsigned __int64 *)&v50, v17, v11, a5, a6);
      if ( v50 != &v52 )
        _libc_free((unsigned __int64)v50);
      goto LABEL_48;
    }
LABEL_20:
    v18 = v43;
    v42[0] = (unsigned __int64)v43;
    v42[1] = 0x300000000LL;
    if ( (_DWORD)v17 )
    {
      sub_2BB7BD0((__int64)v42, (unsigned __int64 *)v41, v17, v11, a5, a6);
      v18 = (_BYTE *)v42[0];
    }
    while ( 2 )
    {
      v19 = (unsigned __int8 *)*((_QWORD *)v18 + 1);
      v20 = 0;
      v21 = *(unsigned __int8 **)(*(v7 - 8) + 8);
      v48 = 1;
      v45 = (unsigned __int8 **)v49;
      v44 = 0;
      v46 = 16;
      v47 = 0;
      v50 = 0;
      v52 = 16;
      v53 = 0;
      v54 = 1;
      v51 = (unsigned __int64)v55;
      v22 = 1;
      while ( 2 )
      {
        if ( v22 )
        {
          v23 = v45;
          v24 = (__int64)&v45[HIDWORD(v46)];
          for ( k = v45; (unsigned __int8 **)v24 != k; ++k )
          {
            if ( v21 == *k )
              goto LABEL_71;
          }
        }
        else if ( sub_C8CA60((__int64)&v44, (__int64)v21) )
        {
          goto LABEL_87;
        }
        if ( !v54 )
        {
          if ( sub_C8CA60((__int64)&v50, (__int64)v19) )
            goto LABEL_87;
LABEL_97:
          if ( v21 == v19 || (unsigned int)qword_500FC48 < v20 )
            goto LABEL_89;
          if ( !v48 )
            goto LABEL_111;
          v38 = v45;
          v24 = HIDWORD(v46);
          v23 = &v45[HIDWORD(v46)];
          if ( v45 != v23 )
          {
            while ( v19 != *v38 )
            {
              if ( v23 == ++v38 )
                goto LABEL_114;
            }
            goto LABEL_104;
          }
LABEL_114:
          if ( HIDWORD(v46) < (unsigned int)v46 )
          {
            v24 = (unsigned int)++HIDWORD(v46);
            *v23 = v19;
            ++v44;
          }
          else
          {
LABEL_111:
            sub_C8CC70((__int64)&v44, (__int64)v19, (__int64)v23, v24, a5, a6);
          }
LABEL_104:
          if ( !v54 )
            goto LABEL_110;
          v39 = (unsigned __int8 **)v51;
          v24 = HIDWORD(v52);
          v23 = (unsigned __int8 **)(v51 + 8LL * HIDWORD(v52));
          if ( (unsigned __int8 **)v51 != v23 )
          {
            while ( v21 != *v39 )
            {
              if ( v23 == ++v39 )
                goto LABEL_112;
            }
            goto LABEL_109;
          }
LABEL_112:
          if ( HIDWORD(v52) < (unsigned int)v52 )
          {
            ++HIDWORD(v52);
            *v23 = v21;
            v50 = (__int64 *)((char *)v50 + 1);
          }
          else
          {
LABEL_110:
            sub_C8CC70((__int64)&v50, (__int64)v21, (__int64)v23, v24, a5, a6);
          }
LABEL_109:
          ++v20;
          v19 = sub_98ACB0(v19, 1u);
          v21 = sub_98ACB0(v21, 1u);
          v22 = v48;
          continue;
        }
        break;
      }
      v37 = (unsigned __int8 **)v51;
      v23 = (unsigned __int8 **)(v51 + 8LL * HIDWORD(v52));
      if ( (unsigned __int8 **)v51 == v23 )
        goto LABEL_97;
      while ( v19 != *v37 )
      {
        if ( v23 == ++v37 )
          goto LABEL_97;
      }
LABEL_87:
      if ( v48 )
      {
        v23 = v45;
        v24 = (__int64)&v45[HIDWORD(v46)];
        if ( v45 == (unsigned __int8 **)v24 )
          goto LABEL_89;
LABEL_71:
        while ( v21 != *v23 )
        {
          if ( ++v23 == (unsigned __int8 **)v24 )
            goto LABEL_89;
        }
      }
      else if ( !sub_C8CA60((__int64)&v44, (__int64)v21) )
      {
LABEL_89:
        v34 = 0;
        goto LABEL_90;
      }
      v34 = v54;
      if ( v54 )
      {
        v35 = (unsigned __int8 **)v51;
        v23 = (unsigned __int8 **)(v51 + 8LL * HIDWORD(v52));
        if ( (unsigned __int8 **)v51 != v23 )
        {
          while ( v19 != *v35 )
          {
            if ( v23 == ++v35 )
              goto LABEL_78;
          }
          v34 = 0;
        }
LABEL_78:
        if ( !v48 )
          goto LABEL_92;
LABEL_79:
        if ( !v34 )
          goto LABEL_93;
LABEL_80:
        v36 = (__int64)v7;
        v7 -= 8;
        sub_2BB7BD0(v36, v7, (__int64)v23, v24, a5, a6);
        v18 = (_BYTE *)v42[0];
        continue;
      }
      break;
    }
    v34 = sub_C8CA60((__int64)&v50, (__int64)v19) == 0;
LABEL_90:
    if ( v54 )
      goto LABEL_78;
    _libc_free(v51);
    if ( v48 )
      goto LABEL_79;
LABEL_92:
    _libc_free((unsigned __int64)v45);
    if ( v34 )
      goto LABEL_80;
LABEL_93:
    sub_2BB7BD0((__int64)v7, v42, (__int64)v23, v24, a5, a6);
    if ( (_BYTE *)v42[0] != v43 )
      _libc_free(v42[0]);
    v29 = v41 + 16;
LABEL_48:
    v41 = v29;
  }
  while ( a2 != v29 );
}
