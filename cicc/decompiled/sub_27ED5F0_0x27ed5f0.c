// Function: sub_27ED5F0
// Address: 0x27ed5f0
//
bool __fastcall sub_27ED5F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  char v7; // si
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  int v17; // r8d
  int v18; // edi
  int v19; // r9d
  char v20; // dl
  __int64 v21; // rax
  int v22; // ecx
  int v23; // edx
  __int64 *v24; // rax
  __int64 *v25; // r12
  __int64 *v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // r9
  __int64 v29; // r10
  __int64 v30; // r8
  unsigned __int64 v31; // rcx
  __int64 *v32; // rdx
  int v33; // esi
  __int64 *v34; // rax
  _QWORD *v35; // rdx
  __int64 v36; // rsi
  __int64 *v37; // rax
  __int64 v38; // r8
  unsigned __int64 v39; // rsi
  __int64 *v40; // rdx
  int v41; // ecx
  __int64 *v42; // rax
  _QWORD *v43; // rdx
  __int64 v44; // rcx
  __int64 *v45; // rax
  unsigned __int64 v46; // rdi
  __int64 *v47; // rcx
  int v48; // esi
  __int64 *v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // rsi
  __int64 *v52; // rax
  __int64 *v53; // rax
  bool v54; // [rsp-89h] [rbp-89h]
  __int64 v55; // [rsp-88h] [rbp-88h] BYREF
  __int64 *v56; // [rsp-80h] [rbp-80h]
  __int64 v57; // [rsp-78h] [rbp-78h]
  int v58; // [rsp-70h] [rbp-70h]
  char v59; // [rsp-6Ch] [rbp-6Ch]
  __int64 v60; // [rsp-68h] [rbp-68h] BYREF

  if ( !(_BYTE)qword_4FFE5C8 )
    return 0;
  v7 = sub_D484B0(*(_QWORD *)(a1 + 16), a2, a3, a4);
  result = 0;
  if ( v7 )
  {
    v9 = *(_QWORD *)(a2 + 40);
    v57 = 8;
    v58 = 0;
    v59 = 1;
    v10 = *(_QWORD *)(v9 + 16);
    v55 = 0;
    v56 = &v60;
    if ( !v10 )
    {
      v17 = 0;
      v18 = 0;
LABEL_33:
      if ( !*(_DWORD *)(a1 + 80) )
        goto LABEL_34;
      v24 = *(__int64 **)(a1 + 72);
      v25 = &v24[2 * *(unsigned int *)(a1 + 88)];
      if ( v24 == v25 )
        goto LABEL_34;
      while ( 1 )
      {
        v26 = v24;
        if ( *v24 != -8192 && *v24 != -4096 )
          break;
        v24 += 2;
        if ( v25 == v24 )
          goto LABEL_34;
      }
      if ( v24 == v25 )
      {
LABEL_34:
        result = v17 == v18;
LABEL_25:
        if ( !v7 )
        {
          v54 = result;
          _libc_free((unsigned __int64)v56);
          return v54;
        }
        return result;
      }
      if ( v24[1] == v9 )
        goto LABEL_50;
      while ( 1 )
      {
        while ( 1 )
        {
          do
          {
            v26 += 2;
            if ( v26 != v25 )
            {
              while ( *v26 == -8192 || *v26 == -4096 )
              {
                v26 += 2;
                if ( v25 == v26 )
                  goto LABEL_47;
              }
              if ( v26 != v25 )
                continue;
            }
LABEL_47:
            v17 = HIDWORD(v57);
            v18 = v58;
            goto LABEL_34;
          }
          while ( v26[1] != v9 );
LABEL_50:
          v27 = *v26;
          v28 = *(_QWORD *)(*v26 - 32);
          if ( v28 )
            break;
          v29 = 0;
LABEL_52:
          v30 = *(_QWORD *)(v27 - 64);
          if ( v9 == v30 && v30 )
          {
            v38 = *(_QWORD *)(v27 + 40);
            if ( v7 )
            {
              v39 = (unsigned __int64)v56;
              v40 = &v56[HIDWORD(v57)];
              v41 = HIDWORD(v57);
              if ( v56 == v40 )
                goto LABEL_68;
              v42 = v56;
              while ( v38 != *v42 )
              {
                if ( v40 == ++v42 )
                  goto LABEL_77;
              }
              --HIDWORD(v57);
              *v42 = v56[HIDWORD(v57)];
              ++v55;
            }
            else
            {
              v52 = sub_C8CA60((__int64)&v55, *(_QWORD *)(v27 + 40));
              if ( v52 )
              {
                *v52 = -2;
                ++v58;
                ++v55;
              }
            }
            v28 = *(_QWORD *)(*v26 - 32);
            if ( v59 )
            {
              v39 = (unsigned __int64)v56;
              v41 = HIDWORD(v57);
              v42 = &v56[HIDWORD(v57)];
              if ( v42 == v56 )
                goto LABEL_68;
LABEL_77:
              v43 = (_QWORD *)v39;
              while ( *v43 != v28 )
              {
                if ( ++v43 == v42 )
                  goto LABEL_68;
              }
              v44 = (unsigned int)(v41 - 1);
              HIDWORD(v57) = v44;
              *v43 = *(_QWORD *)(v39 + 8 * v44);
              v7 = v59;
              ++v55;
            }
            else
            {
              v37 = sub_C8CA60((__int64)&v55, *(_QWORD *)(*v26 - 32));
              if ( v37 )
                goto LABEL_67;
LABEL_68:
              v7 = v59;
            }
          }
          else
          {
            if ( v7 )
            {
              v31 = (unsigned __int64)v56;
              v32 = &v56[HIDWORD(v57)];
              v33 = HIDWORD(v57);
              if ( v56 == v32 )
                goto LABEL_68;
              v34 = v56;
              while ( *v34 != v29 )
              {
                if ( v32 == ++v34 )
                  goto LABEL_62;
              }
              --HIDWORD(v57);
              *v34 = v56[HIDWORD(v57)];
              ++v55;
            }
            else
            {
              v45 = sub_C8CA60((__int64)&v55, v29);
              if ( v45 )
              {
                *v45 = -2;
                ++v58;
                ++v55;
              }
            }
            v30 = *(_QWORD *)(*v26 - 64);
            if ( !v59 )
            {
LABEL_66:
              v37 = sub_C8CA60((__int64)&v55, v30);
              if ( !v37 )
                goto LABEL_68;
LABEL_67:
              *v37 = -2;
              ++v58;
              ++v55;
              goto LABEL_68;
            }
            v31 = (unsigned __int64)v56;
            v33 = HIDWORD(v57);
            v34 = &v56[HIDWORD(v57)];
            if ( v34 == v56 )
              goto LABEL_68;
LABEL_62:
            v35 = (_QWORD *)v31;
            while ( *v35 != v30 )
            {
              if ( ++v35 == v34 )
                goto LABEL_68;
            }
            v36 = (unsigned int)(v33 - 1);
            HIDWORD(v57) = v36;
            *v35 = *(_QWORD *)(v31 + 8 * v36);
            v7 = v59;
            ++v55;
          }
        }
        v29 = *(_QWORD *)(*v26 - 32);
        if ( v9 != v28 )
          goto LABEL_52;
        if ( v7 )
        {
          v46 = (unsigned __int64)v56;
          v47 = &v56[HIDWORD(v57)];
          v48 = HIDWORD(v57);
          if ( v56 == v47 )
          {
LABEL_101:
            v30 = *(_QWORD *)(v27 - 64);
            goto LABEL_91;
          }
          v49 = v56;
          while ( *(_QWORD *)(v27 + 40) != *v49 )
          {
            if ( v47 == ++v49 )
              goto LABEL_101;
          }
          --HIDWORD(v57);
          *v49 = v56[HIDWORD(v57)];
          ++v55;
        }
        else
        {
          v53 = sub_C8CA60((__int64)&v55, *(_QWORD *)(v27 + 40));
          if ( v53 )
          {
            *v53 = -2;
            ++v58;
            ++v55;
          }
        }
        v30 = *(_QWORD *)(*v26 - 64);
        if ( !v59 )
          goto LABEL_66;
        v46 = (unsigned __int64)v56;
        v48 = HIDWORD(v57);
        v47 = &v56[HIDWORD(v57)];
LABEL_91:
        v50 = (_QWORD *)v46;
        if ( (__int64 *)v46 == v47 )
          goto LABEL_68;
        while ( *v50 != v30 )
        {
          if ( ++v50 == v47 )
            goto LABEL_68;
        }
        v51 = (unsigned int)(v48 - 1);
        HIDWORD(v57) = v51;
        *v50 = *(_QWORD *)(v46 + 8 * v51);
        v7 = v59;
        ++v55;
      }
    }
    v11 = v10;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v11 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v12 - 30) <= 0xAu )
        break;
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
      {
        v19 = 0;
        v17 = 0;
        v18 = 0;
        goto LABEL_18;
      }
    }
    v13 = *(_QWORD *)(v12 + 40);
LABEL_8:
    v14 = v56;
    v15 = HIDWORD(v57);
    v16 = (__int64)&v56[HIDWORD(v57)];
    if ( v56 == (__int64 *)v16 )
    {
LABEL_30:
      if ( HIDWORD(v57) >= (unsigned int)v57 )
      {
        while ( 1 )
        {
          sub_C8CC70((__int64)&v55, v13, v16, v15, v13, v8);
          v11 = *(_QWORD *)(v11 + 8);
          v7 = v59;
          if ( !v11 )
            break;
LABEL_13:
          v16 = *(_QWORD *)(v11 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v16 - 30) > 0xAu )
            goto LABEL_12;
          v13 = *(_QWORD *)(v16 + 40);
          if ( v7 )
            goto LABEL_8;
        }
LABEL_16:
        v17 = HIDWORD(v57);
        v18 = v58;
        v10 = *(_QWORD *)(v9 + 16);
        v19 = HIDWORD(v57) - v58;
        do
        {
          if ( !v10 )
          {
            v23 = 0;
            goto LABEL_24;
          }
LABEL_18:
          v20 = **(_BYTE **)(v10 + 24);
          v21 = v10;
          v10 = *(_QWORD *)(v10 + 8);
        }
        while ( (unsigned __int8)(v20 - 30) > 0xAu );
        v22 = 0;
        while ( 1 )
        {
          v21 = *(_QWORD *)(v21 + 8);
          if ( !v21 )
            break;
          while ( (unsigned __int8)(**(_BYTE **)(v21 + 24) - 30) <= 0xAu )
          {
            v21 = *(_QWORD *)(v21 + 8);
            ++v22;
            if ( !v21 )
              goto LABEL_23;
          }
        }
LABEL_23:
        v23 = v22 + 1;
LABEL_24:
        result = 0;
        if ( v19 != v23 )
          goto LABEL_25;
        goto LABEL_33;
      }
      v15 = (unsigned int)++HIDWORD(v57);
      *(_QWORD *)v16 = v13;
      v7 = v59;
      ++v55;
    }
    else
    {
      while ( v13 != *v14 )
      {
        if ( (__int64 *)v16 == ++v14 )
          goto LABEL_30;
      }
    }
LABEL_12:
    v11 = *(_QWORD *)(v11 + 8);
    if ( v11 )
      goto LABEL_13;
    goto LABEL_16;
  }
  return result;
}
