// Function: sub_2D5DA80
// Address: 0x2d5da80
//
__int64 __fastcall sub_2D5DA80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        _DWORD *a8)
{
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  unsigned int v12; // eax
  __int64 v13; // r12
  char v14; // al
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  void (__fastcall *v22)(unsigned __int64 *, __int64, __int64, __int64, __int64); // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // r14
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rbx
  unsigned __int64 v28; // rdi
  __int64 v29; // r13
  unsigned __int64 v30; // r12
  __int64 v31; // r14
  unsigned __int64 v32; // r15
  unsigned __int64 *v33; // r14
  unsigned __int64 *v34; // r13
  unsigned __int64 *v35; // r12
  __int64 v36; // [rsp+0h] [rbp-B0h]
  __int64 v37; // [rsp+8h] [rbp-A8h]
  __int64 v38; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v39; // [rsp+10h] [rbp-A0h]
  int v40; // [rsp+20h] [rbp-90h]
  unsigned __int8 v44; // [rsp+46h] [rbp-6Ah]
  unsigned __int8 v45; // [rsp+47h] [rbp-69h]
  unsigned __int64 v47; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v48; // [rsp+58h] [rbp-58h]
  unsigned __int8 v49; // [rsp+70h] [rbp-40h]

  v40 = a3;
  sub_BED950((__int64)&v47, a3, a1);
  v45 = v49;
  if ( !v49 )
    return 0;
  switch ( *(_BYTE *)a1 )
  {
    case '*':
    case '?':
    case 'L':
    case 'M':
      goto LABEL_4;
    case '.':
    case '6':
      if ( **(_BYTE **)(sub_986520(a1) + 32) == 17 )
        goto LABEL_4;
      return v45;
    case 'N':
    case 'O':
      v20 = *(_QWORD *)(a1 + 8);
      if ( v20 == *(_QWORD *)(*(_QWORD *)sub_986520(a1) + 8LL) || (*(_BYTE *)(v20 + 8) & 0xFD) != 0xC )
        return v45;
LABEL_4:
      v11 = *(_QWORD *)(a1 + 16);
      if ( !v11 )
        return 0;
      break;
    default:
      return v45;
  }
LABEL_5:
  v12 = (*a8)++;
  if ( v12 >= (unsigned int)qword_5016AE8 )
    return v45;
  v13 = *(_QWORD *)(v11 + 24);
  v14 = *(_BYTE *)v13;
  if ( *(_BYTE *)v13 == 61 )
  {
    v15 = *(_QWORD *)(v13 + 8);
LABEL_8:
    v16 = *(unsigned int *)(a2 + 8);
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v16 + 1, 0x10u, v9, v10);
      v16 = *(unsigned int *)(a2 + 8);
    }
    v17 = (__int64 *)(*(_QWORD *)a2 + 16 * v16);
    *v17 = v11;
    v17[1] = v15;
    ++*(_DWORD *)(a2 + 8);
    goto LABEL_11;
  }
  switch ( v14 )
  {
    case '>':
      if ( (unsigned int)sub_BD2910(v11) != 1 )
        return v45;
      goto LABEL_16;
    case 'B':
      if ( (unsigned int)sub_BD2910(v11) )
        return v45;
      v19 = *(_QWORD *)(v13 - 32);
      goto LABEL_20;
    case 'A':
      if ( (unsigned int)sub_BD2910(v11) )
        return v45;
LABEL_16:
      v19 = *(_QWORD *)(v13 - 64);
LABEL_20:
      v15 = *(_QWORD *)(v19 + 8);
      goto LABEL_8;
    case 'U':
      if ( ((unsigned __int8)sub_A73ED0((_QWORD *)(v13 + 72), 5) || (unsigned __int8)sub_B49560(v13, 5))
        && !sub_11F3070(*(_QWORD *)(v13 + 40), a6, a7) )
      {
        goto LABEL_11;
      }
      if ( **(_BYTE **)(v13 - 32) != 25 )
        return v45;
      v21 = sub_B43CB0(v13);
      v22 = *(void (__fastcall **)(unsigned __int64 *, __int64, __int64, __int64, __int64))(*(_QWORD *)a4 + 2456LL);
      v23 = sub_B2BEC0(v21);
      v22(&v47, a4, v23, a5, v13);
      v24 = v48;
      if ( v47 == v48 )
      {
        v44 = v45;
        if ( v48 )
          goto LABEL_64;
        goto LABEL_11;
      }
      v38 = v11;
      v25 = v47;
      v26 = v48;
      while ( 1 )
      {
        (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a4 + 2480LL))(
          a4,
          v25,
          0,
          0,
          0);
        if ( a1 == *(_QWORD *)(v25 + 232) )
        {
          if ( *(_DWORD *)(v25 + 224) != 2 )
          {
            v44 = 0;
            v11 = v38;
LABEL_38:
            v27 = v47;
            v39 = v48;
            v24 = v47;
            if ( v48 != v47 )
            {
              v37 = v11;
              v36 = a2;
              do
              {
                v28 = *(_QWORD *)(v27 + 192);
                if ( v28 != v27 + 208 )
                  j_j___libc_free_0(v28);
                v29 = *(_QWORD *)(v27 + 64);
                v30 = v29 + 56LL * *(unsigned int *)(v27 + 72);
                if ( v29 != v30 )
                {
                  do
                  {
                    v31 = *(unsigned int *)(v30 - 40);
                    v32 = *(_QWORD *)(v30 - 48);
                    v30 -= 56LL;
                    v33 = (unsigned __int64 *)(v32 + 32 * v31);
                    if ( (unsigned __int64 *)v32 != v33 )
                    {
                      do
                      {
                        v33 -= 4;
                        if ( (unsigned __int64 *)*v33 != v33 + 2 )
                          j_j___libc_free_0(*v33);
                      }
                      while ( (unsigned __int64 *)v32 != v33 );
                      v32 = *(_QWORD *)(v30 + 8);
                    }
                    if ( v32 != v30 + 24 )
                      _libc_free(v32);
                  }
                  while ( v29 != v30 );
                  v30 = *(_QWORD *)(v27 + 64);
                }
                if ( v30 != v27 + 80 )
                  _libc_free(v30);
                v34 = *(unsigned __int64 **)(v27 + 16);
                v35 = &v34[4 * *(unsigned int *)(v27 + 24)];
                if ( v34 != v35 )
                {
                  do
                  {
                    v35 -= 4;
                    if ( (unsigned __int64 *)*v35 != v35 + 2 )
                      j_j___libc_free_0(*v35);
                  }
                  while ( v34 != v35 );
                  v35 = *(unsigned __int64 **)(v27 + 16);
                }
                if ( v35 != (unsigned __int64 *)(v27 + 32) )
                  _libc_free((unsigned __int64)v35);
                v27 += 248LL;
              }
              while ( v39 != v27 );
              v11 = v37;
              a2 = v36;
              v24 = v47;
            }
            if ( v24 )
LABEL_64:
              j_j___libc_free_0(v24);
            if ( !v44 )
              return v45;
LABEL_11:
            v11 = *(_QWORD *)(v11 + 8);
            if ( !v11 )
              return 0;
            goto LABEL_5;
          }
          if ( !*(_BYTE *)(v25 + 10) )
          {
            v44 = 0;
            v11 = v38;
            goto LABEL_38;
          }
        }
        v25 += 248LL;
        if ( v26 == v25 )
        {
          v11 = v38;
          v44 = v45;
          goto LABEL_38;
        }
      }
  }
  if ( !(unsigned __int8)sub_2D5DA80(v13, a2, v40, a4, a5, a6, (__int64)a7, (__int64)a8) )
    goto LABEL_11;
  return v45;
}
