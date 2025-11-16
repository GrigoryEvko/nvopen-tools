// Function: sub_1D5CA40
// Address: 0x1d5ca40
//
__int64 __fastcall sub_1D5CA40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int8 v7; // dl
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v12; // r12
  char v13; // al
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 *v18; // rax
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 v23; // r12
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  void (__fastcall *v30)(__int64 *, __int64, __int64, __int64, __int64); // r14
  __int64 v31; // rax
  __int64 v32; // r14
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 v36; // r13
  __int64 v37; // rdi
  __int64 v38; // r15
  unsigned __int64 v39; // r12
  __int64 v40; // rbx
  unsigned __int64 v41; // r13
  _QWORD *v42; // rbx
  _QWORD *v43; // r15
  _QWORD *v44; // r12
  __int64 v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+10h] [rbp-90h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  unsigned __int8 v48; // [rsp+18h] [rbp-88h]
  __int64 v49; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+20h] [rbp-80h]
  unsigned int v55; // [rsp+48h] [rbp-58h]
  char v56; // [rsp+4Eh] [rbp-52h]
  unsigned __int8 v57; // [rsp+4Fh] [rbp-51h]
  __int64 v58; // [rsp+50h] [rbp-50h] BYREF
  __int64 v59; // [rsp+58h] [rbp-48h]
  __int64 v60; // [rsp+60h] [rbp-40h]

  sub_1412190(a3, a1);
  v57 = v7;
  if ( !v7 )
    return 0;
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case '#':
    case '8':
    case 'E':
    case 'F':
      goto LABEL_6;
    case '\'':
    case '/':
      if ( *(_BYTE *)(*(_QWORD *)(sub_13CF970(a1) + 24) + 16LL) == 13 )
        goto LABEL_6;
      return v57;
    case 'G':
    case 'H':
      v23 = *(_QWORD *)a1;
      if ( v23 == **(_QWORD **)sub_13CF970(a1) || (*(_BYTE *)(v23 + 8) & 0xFB) != 0xB )
        return v57;
LABEL_6:
      v9 = sub_15F2060(a1) + 112;
      v56 = sub_1560180(v9, 34);
      if ( !v56 )
        v56 = sub_1560180(v9, 17);
      v10 = *(_QWORD *)(a1 + 8);
      if ( !v10 )
        return 0;
      v55 = a6 + 1;
      if ( a6 > 19 )
        return v57;
      break;
    default:
      return v57;
  }
LABEL_10:
  v12 = (__int64)sub_1648700(v10);
  v13 = *(_BYTE *)(v12 + 16);
  if ( v13 == 54 )
  {
    v16 = (unsigned int)sub_1648720(v10);
    v17 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v17 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v14, v15);
      v17 = *(unsigned int *)(a2 + 8);
    }
    v18 = (__int64 *)(*(_QWORD *)a2 + 16 * v17);
    *v18 = v12;
    v18[1] = v16;
    ++*(_DWORD *)(a2 + 8);
    goto LABEL_14;
  }
  if ( v13 == 55 )
  {
    if ( (unsigned int)sub_1648720(v10) == 1 )
    {
      v21 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v21 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v19, v20);
        v21 = *(unsigned int *)(a2 + 8);
      }
      v22 = (__int64 *)(*(_QWORD *)a2 + 16 * v21);
      *v22 = v12;
      v22[1] = 1;
      ++*(_DWORD *)(a2 + 8);
      goto LABEL_14;
    }
    return v57;
  }
  if ( v13 != 59 && v13 != 58 )
  {
    if ( v13 != 78 )
    {
      if ( !(unsigned __int8)sub_1D5CA40(v12, a2, a3, a4, a5, v55) )
        goto LABEL_14;
      return v57;
    }
    if ( !v56 )
    {
      if ( (unsigned __int8)sub_1560260((_QWORD *)(v12 + 56), -1, 7) )
        goto LABEL_14;
      v28 = *(_QWORD *)(v12 - 24);
      if ( *(_BYTE *)(v28 + 16) )
        goto LABEL_34;
      v58 = *(_QWORD *)(v28 + 112);
      if ( (unsigned __int8)sub_1560260(&v58, -1, 7) )
        goto LABEL_14;
    }
    v28 = *(_QWORD *)(v12 - 24);
LABEL_34:
    if ( *(_BYTE *)(v28 + 16) != 20 )
      return v57;
    v29 = sub_15F2060(v12);
    v30 = *(void (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(*(_QWORD *)a4 + 1352LL);
    v31 = sub_1632FA0(*(_QWORD *)(v29 + 40));
    v30(&v58, a4, v31, a5, v12 | 4);
    v32 = v58;
    v49 = v59;
    v33 = 0xEF7BDEF7BDEF7BDFLL * ((v59 - v58) >> 3);
    if ( !(_DWORD)v33 )
    {
LABEL_78:
      v48 = v57;
      goto LABEL_42;
    }
    v47 = v10;
    v50 = 248LL * (unsigned int)v33;
    v34 = v58;
    v35 = 0;
    while ( 1 )
    {
      v36 = v34 + v35;
      (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a4 + 1376LL))(
        a4,
        v34 + v35,
        0,
        0,
        0);
      if ( a1 == *(_QWORD *)(v36 + 232) )
      {
        if ( *(_DWORD *)(v36 + 224) != 2 )
        {
          v10 = v47;
          v48 = 0;
          v32 = v58;
          v49 = v59;
LABEL_42:
          if ( v49 != v32 )
          {
            v46 = v10;
            v45 = a2;
            do
            {
              v37 = *(_QWORD *)(v32 + 192);
              if ( v37 != v32 + 208 )
                j_j___libc_free_0(v37, *(_QWORD *)(v32 + 208) + 1LL);
              v38 = *(_QWORD *)(v32 + 64);
              v39 = v38 + 56LL * *(unsigned int *)(v32 + 72);
              if ( v38 != v39 )
              {
                do
                {
                  v40 = *(unsigned int *)(v39 - 40);
                  v41 = *(_QWORD *)(v39 - 48);
                  v39 -= 56LL;
                  v42 = (_QWORD *)(v41 + 32 * v40);
                  if ( (_QWORD *)v41 != v42 )
                  {
                    do
                    {
                      v42 -= 4;
                      if ( (_QWORD *)*v42 != v42 + 2 )
                        j_j___libc_free_0(*v42, v42[2] + 1LL);
                    }
                    while ( (_QWORD *)v41 != v42 );
                    v41 = *(_QWORD *)(v39 + 8);
                  }
                  if ( v41 != v39 + 24 )
                    _libc_free(v41);
                }
                while ( v38 != v39 );
                v39 = *(_QWORD *)(v32 + 64);
              }
              if ( v39 != v32 + 80 )
                _libc_free(v39);
              v43 = *(_QWORD **)(v32 + 16);
              v44 = &v43[4 * *(unsigned int *)(v32 + 24)];
              if ( v43 != v44 )
              {
                do
                {
                  v44 -= 4;
                  if ( (_QWORD *)*v44 != v44 + 2 )
                    j_j___libc_free_0(*v44, v44[2] + 1LL);
                }
                while ( v43 != v44 );
                v44 = *(_QWORD **)(v32 + 16);
              }
              if ( v44 != (_QWORD *)(v32 + 32) )
                _libc_free((unsigned __int64)v44);
              v32 += 248;
            }
            while ( v49 != v32 );
            v10 = v46;
            a2 = v45;
          }
          if ( v58 )
            j_j___libc_free_0(v58, v60 - v58);
          if ( !v48 )
            return v57;
LABEL_14:
          v10 = *(_QWORD *)(v10 + 8);
          if ( !v10 )
            return 0;
          if ( ++v55 == 21 )
            return v57;
          goto LABEL_10;
        }
        if ( !*(_BYTE *)(v36 + 10) )
        {
          v10 = v47;
          v32 = v58;
          v48 = 0;
          v49 = v59;
          goto LABEL_42;
        }
      }
      v34 = v58;
      v35 += 248;
      if ( v35 == v50 )
      {
        v32 = v58;
        v10 = v47;
        v49 = v59;
        goto LABEL_78;
      }
    }
  }
  if ( !(unsigned int)sub_1648720(v10) )
  {
    v26 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v26 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v24, v25);
      v26 = *(unsigned int *)(a2 + 8);
    }
    v27 = (__int64 *)(*(_QWORD *)a2 + 16 * v26);
    *v27 = v12;
    v27[1] = 0;
    ++*(_DWORD *)(a2 + 8);
    goto LABEL_14;
  }
  return v57;
}
