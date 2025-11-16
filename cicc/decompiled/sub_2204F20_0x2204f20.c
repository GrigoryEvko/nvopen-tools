// Function: sub_2204F20
// Address: 0x2204f20
//
__int64 __fastcall sub_2204F20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rbx
  __int64 i; // rbx
  __int64 v13; // rcx
  __int64 v14; // r12
  __int64 v15; // r15
  int v16; // ebx
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // rax
  __int64 j; // rbx
  __int64 *v21; // rax
  __int64 *v22; // r8
  __int64 v23; // r14
  __int64 *v24; // r15
  __int64 *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned __int16 *v29; // rdx
  int v30; // eax
  __int64 v31; // rax
  int v32; // r14d
  __int64 *v33; // rdi
  __int64 *v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 *v40; // r14
  __int64 *v41; // r15
  __int64 *v42; // rax
  unsigned __int64 *v43; // rdi
  unsigned __int64 v44; // rdx
  __int64 v45; // rcx
  int v46; // r8d
  int v47; // r9d
  _QWORD *v48; // r12
  __int64 *v49; // rdx
  _QWORD *v51; // [rsp+10h] [rbp-1A0h]
  _QWORD *v52; // [rsp+18h] [rbp-198h]
  __int64 *v53; // [rsp+18h] [rbp-198h]
  _QWORD *v54; // [rsp+20h] [rbp-190h]
  unsigned __int64 v55; // [rsp+28h] [rbp-188h]
  unsigned __int8 v56; // [rsp+37h] [rbp-179h]
  __int64 *v57; // [rsp+40h] [rbp-170h] BYREF
  unsigned int v58; // [rsp+48h] [rbp-168h]
  unsigned int v59; // [rsp+4Ch] [rbp-164h]
  _BYTE v60[128]; // [rsp+50h] [rbp-160h] BYREF
  __int64 v61; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 *v62; // [rsp+D8h] [rbp-D8h]
  __int64 *v63; // [rsp+E0h] [rbp-D0h]
  __int64 v64; // [rsp+E8h] [rbp-C8h]
  int v65; // [rsp+F0h] [rbp-C0h]
  _BYTE v66[184]; // [rsp+F8h] [rbp-B8h] BYREF

  v6 = (__int64 *)v66;
  v7 = *(_QWORD *)(a2 + 320);
  v61 = 0;
  v8 = *(_QWORD *)(a2 + 40);
  v62 = (__int64 *)v66;
  v63 = (__int64 *)v66;
  v64 = 16;
  v65 = 0;
  v51 = (_QWORD *)(a2 + 320);
  v52 = (_QWORD *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
  if ( a2 + 320 != (v7 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v56 = 0;
LABEL_3:
    v9 = v52[3];
    v54 = v52 + 3;
    v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      BUG();
    v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v10 & 4) == 0 && (*(_BYTE *)(v10 + 46) & 4) != 0 )
    {
      for ( i = *(_QWORD *)v10; ; i = *(_QWORD *)v11 )
      {
        v11 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v11 + 46) & 4) == 0 )
          break;
      }
    }
LABEL_9:
    if ( v54 == (_QWORD *)v11 )
      goto LABEL_24;
LABEL_10:
    v13 = *(_QWORD *)(v11 + 32);
    v14 = v13 + 40LL * *(unsigned int *)(v11 + 40);
    if ( v13 == v14 )
      goto LABEL_17;
    v55 = v11;
    v15 = *(_QWORD *)(v11 + 32);
LABEL_12:
    if ( *(_BYTE *)v15 )
      goto LABEL_15;
    if ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 )
      goto LABEL_15;
    v16 = *(_DWORD *)(v15 + 8);
    if ( v16 >= 0 )
      goto LABEL_15;
    while ( 1 )
    {
      v27 = sub_1E69D00(v8, v16);
      v28 = v27;
      if ( !v27
        || (v29 = *(unsigned __int16 **)(v27 + 16), v30 = *v29, (_WORD)v30 != 15)
        && (unsigned int)(v30 - 4883) > 1
        && (v29[8] & 0x10) == 0
        || (v31 = *(_QWORD *)(v28 + 32), *(_BYTE *)(v31 + 40))
        || (v32 = *(_DWORD *)(v31 + 48), v32 >= 0) )
      {
        if ( v16 != *(_DWORD *)(v15 + 8) )
        {
          sub_1E310D0(v15, v16);
          v56 = 1;
        }
LABEL_15:
        v15 += 40;
        if ( v14 == v15 )
        {
          v11 = v55;
LABEL_17:
          v17 = (_QWORD *)(*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL);
          v18 = v17;
          if ( !v17 )
            BUG();
          v11 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
          v19 = *v17;
          if ( (v19 & 4) != 0 || (*((_BYTE *)v18 + 46) & 4) == 0 )
            goto LABEL_9;
          for ( j = v19; ; j = *(_QWORD *)v11 )
          {
            v11 = j & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v11 + 46) & 4) == 0 )
              break;
          }
          if ( v54 == (_QWORD *)v11 )
          {
LABEL_24:
            v52 = (_QWORD *)(*v52 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v51 == v52 )
            {
              v21 = v63;
              v6 = v62;
              goto LABEL_26;
            }
            goto LABEL_3;
          }
          goto LABEL_10;
        }
        goto LABEL_12;
      }
      v26 = v62;
      if ( v63 != v62 )
        break;
      v33 = &v62[HIDWORD(v64)];
      if ( v62 == v33 )
      {
LABEL_55:
        if ( HIDWORD(v64) >= (unsigned int)v64 )
          break;
        ++HIDWORD(v64);
        *v33 = v28;
        ++v61;
      }
      else
      {
        v34 = 0;
        while ( v28 != *v26 )
        {
          if ( *v26 == -2 )
            v34 = v26;
          if ( v33 == ++v26 )
          {
            if ( !v34 )
              goto LABEL_55;
            *v34 = v28;
            --v65;
            ++v61;
            break;
          }
        }
      }
LABEL_38:
      v16 = v32;
    }
    sub_16CCBA0((__int64)&v61, v28);
    goto LABEL_38;
  }
  v56 = 0;
  v21 = (__int64 *)v66;
LABEL_26:
  v59 = 16;
  v57 = (__int64 *)v60;
  v58 = 0;
  if ( v6 == v21 )
    goto LABEL_92;
LABEL_27:
  v22 = &v21[(unsigned int)v64];
LABEL_28:
  if ( v22 != v21 )
  {
    while ( 1 )
    {
      v23 = *v21;
      v24 = v21;
      if ( (unsigned __int64)*v21 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v22 == ++v21 )
        goto LABEL_31;
    }
    if ( v21 != v22 )
    {
      v35 = 0;
      v36 = *(unsigned int *)(*(_QWORD *)(v23 + 32) + 8LL);
      if ( (int)v36 >= 0 )
      {
LABEL_59:
        v37 = *(_QWORD *)(*(_QWORD *)(v8 + 272) + 8 * v36);
        goto LABEL_60;
      }
      while ( 1 )
      {
        v37 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 16 * (v36 & 0x7FFFFFFF) + 8);
LABEL_60:
        if ( !v37 )
          goto LABEL_79;
        if ( (*(_BYTE *)(v37 + 3) & 0x10) != 0 )
          break;
LABEL_62:
        v38 = v24 + 1;
        if ( v24 + 1 == v22 )
          goto LABEL_65;
        while ( 1 )
        {
          v23 = *v38;
          v24 = v38;
          if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v22 == ++v38 )
            goto LABEL_65;
        }
        if ( v22 == v38 )
        {
LABEL_65:
          v39 = (unsigned int)v35;
          v40 = &v57[v39];
          if ( v57 == &v57[v39] )
            goto LABEL_90;
          v41 = v57;
          while ( 2 )
          {
            v48 = (_QWORD *)*v41;
            v42 = v62;
            if ( v63 == v62 )
            {
              v49 = &v62[HIDWORD(v64)];
              if ( v62 == v49 )
              {
LABEL_87:
                v42 = &v62[HIDWORD(v64)];
              }
              else
              {
                while ( v48 != (_QWORD *)*v42 )
                {
                  if ( v49 == ++v42 )
                    goto LABEL_87;
                }
              }
LABEL_75:
              if ( v42 != v49 )
              {
                *v42 = -2;
                ++v65;
              }
            }
            else
            {
              v42 = sub_16CC9F0((__int64)&v61, *v41);
              if ( v48 == (_QWORD *)*v42 )
              {
                if ( v63 == v62 )
                  v49 = &v63[HIDWORD(v64)];
                else
                  v49 = &v63[(unsigned int)v64];
                goto LABEL_75;
              }
              if ( v63 == v62 )
              {
                v42 = &v63[HIDWORD(v64)];
                v49 = v42;
                goto LABEL_75;
              }
            }
            ++v41;
            sub_1DD5BC0(v48[3] + 16LL, (__int64)v48);
            v43 = (unsigned __int64 *)v48[1];
            v44 = *v48 & 0xFFFFFFFFFFFFFFF8LL;
            *v43 = v44 | *v43 & 7;
            *(_QWORD *)(v44 + 8) = v43;
            *v48 &= 7uLL;
            v48[1] = 0;
            sub_1E0A0F0(a2, (__int64)v48, v44, v45, v46, v47);
            if ( v40 != v41 )
              continue;
            break;
          }
          v56 = 1;
          LODWORD(v35) = v58;
LABEL_90:
          if ( (_DWORD)v35 )
          {
            v21 = v63;
            v58 = 0;
            if ( v62 == v63 )
            {
LABEL_92:
              v22 = &v21[HIDWORD(v64)];
              goto LABEL_28;
            }
            goto LABEL_27;
          }
          goto LABEL_31;
        }
        v36 = *(unsigned int *)(*(_QWORD *)(v23 + 32) + 8LL);
        if ( (int)v36 >= 0 )
          goto LABEL_59;
      }
      while ( 1 )
      {
        v37 = *(_QWORD *)(v37 + 32);
        if ( !v37 )
          break;
        if ( (*(_BYTE *)(v37 + 3) & 0x10) == 0 )
          goto LABEL_62;
      }
LABEL_79:
      if ( v59 <= (unsigned int)v35 )
      {
        v53 = v22;
        sub_16CD150((__int64)&v57, v60, 0, 8, (int)v22, a6);
        v35 = v58;
        v22 = v53;
      }
      v57[v35] = v23;
      v35 = ++v58;
      goto LABEL_62;
    }
  }
LABEL_31:
  if ( v57 != (__int64 *)v60 )
    _libc_free((unsigned __int64)v57);
  if ( v63 != v62 )
    _libc_free((unsigned __int64)v63);
  return v56;
}
