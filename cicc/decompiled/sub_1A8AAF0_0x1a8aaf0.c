// Function: sub_1A8AAF0
// Address: 0x1a8aaf0
//
void __fastcall sub_1A8AAF0(_QWORD *a1)
{
  __int64 v1; // r15
  _QWORD *v3; // r13
  char v4; // al
  _QWORD *v5; // rax
  unsigned int v6; // eax
  _QWORD *v7; // rdx
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  _QWORD *v15; // r13
  __int64 v16; // rbx
  unsigned __int8 v17; // si
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // r15
  __int64 v21; // rdx
  _QWORD *v22; // r12
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  unsigned int v25; // eax
  _QWORD *v26; // rdx
  unsigned __int64 v27; // rsi
  __int64 v28; // rax
  _QWORD *v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned int v34; // eax
  _QWORD *v35; // r12
  _QWORD *v36; // rbx
  __int64 v37; // rsi
  __int64 v38; // rax
  _QWORD *v39; // r12
  _QWORD *v40; // r15
  __int64 v41; // rsi
  _QWORD *v42; // [rsp+0h] [rbp-B0h]
  _QWORD *v43; // [rsp+10h] [rbp-A0h]
  _QWORD *v44; // [rsp+18h] [rbp-98h]
  _QWORD *v45; // [rsp+18h] [rbp-98h]
  __int64 v46; // [rsp+18h] [rbp-98h]
  _QWORD *v47; // [rsp+18h] [rbp-98h]
  _QWORD *v48; // [rsp+18h] [rbp-98h]
  __int64 v49; // [rsp+28h] [rbp-88h] BYREF
  __int64 v50; // [rsp+30h] [rbp-80h]
  __int64 v51; // [rsp+38h] [rbp-78h]
  __int64 v52; // [rsp+40h] [rbp-70h]
  void *v53; // [rsp+50h] [rbp-60h]
  __int64 v54; // [rsp+58h] [rbp-58h] BYREF
  __int64 v55; // [rsp+60h] [rbp-50h]
  __int64 v56; // [rsp+68h] [rbp-48h]
  __int64 v57; // [rsp+70h] [rbp-40h]

  v1 = 0;
  v3 = (_QWORD *)*a1;
  if ( (_QWORD *)*a1 != a1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = *((_BYTE *)v3 + 120) ^ 1;
        if ( !v1 )
          break;
        if ( !v4 )
          goto LABEL_7;
        sub_1A898D0((__int64)(v3 + 2), v1);
        v5 = (_QWORD *)*v3;
        --a1[2];
        v43 = v5;
        sub_2208CA0(v3);
        if ( *((_BYTE *)v3 + 288) )
        {
          v34 = *((_DWORD *)v3 + 70);
          if ( v34 )
          {
            v35 = (_QWORD *)v3[33];
            v36 = &v35[2 * v34];
            do
            {
              if ( *v35 != -8 && *v35 != -4 )
              {
                v37 = v35[1];
                if ( v37 )
                  sub_161E7C0((__int64)(v35 + 1), v37);
              }
              v35 += 2;
            }
            while ( v36 != v35 );
          }
          j___libc_free_0(v3[33]);
        }
        v6 = *((_DWORD *)v3 + 62);
        if ( v6 )
        {
          v7 = (_QWORD *)v3[29];
          v49 = 2;
          v8 = (unsigned __int64)v6 << 6;
          v50 = 0;
          v9 = -8;
          v51 = -8;
          v10 = (_QWORD *)((char *)v7 + v8);
          v52 = 0;
          v54 = 2;
          v55 = 0;
          v56 = -16;
          v53 = &unk_49E6B50;
          v57 = 0;
          while ( 1 )
          {
            v11 = v7[3];
            if ( v11 != v9 )
            {
              v9 = v56;
              if ( v11 != v56 )
              {
                v12 = v7[7];
                if ( v12 != 0 && v12 != -8 && v12 != -16 )
                {
                  v44 = v7;
                  sub_1649B30(v7 + 5);
                  v7 = v44;
                  v11 = v44[3];
                }
                v9 = v11;
              }
            }
            *v7 = &unk_49EE2B0;
            if ( v9 != 0 && v9 != -8 && v9 != -16 )
            {
              v45 = v7;
              sub_1649B30(v7 + 1);
              v7 = v45;
            }
            v7 += 8;
            if ( v10 == v7 )
              break;
            v9 = v51;
          }
          v53 = &unk_49EE2B0;
          if ( v56 != 0 && v56 != -8 && v56 != -16 )
            sub_1649B30(&v54);
          if ( v51 != 0 && v51 != -8 && v51 != -16 )
            sub_1649B30(&v49);
        }
        j___libc_free_0(v3[29]);
        v13 = v3[18];
        if ( (_QWORD *)v13 != v3 + 20 )
          _libc_free(v13);
        v14 = v3[4];
        if ( v14 != v3[3] )
          _libc_free(v14);
        j_j___libc_free_0(v3, 304);
        v3 = v43;
LABEL_4:
        if ( a1 == v3 )
          goto LABEL_8;
      }
      if ( v4 )
      {
        v1 = (__int64)(v3 + 2);
        v3 = (_QWORD *)*v3;
        goto LABEL_4;
      }
LABEL_7:
      v3 = (_QWORD *)*v3;
      v1 = 0;
      if ( a1 == v3 )
      {
LABEL_8:
        if ( byte_4FB5600 )
          return;
        v15 = (_QWORD *)*a1;
        if ( (_QWORD *)*a1 == a1 )
          return;
        v16 = 0;
        while ( 2 )
        {
          v17 = *((_BYTE *)v15 + 120);
          v46 = (__int64)(v15 + 2);
          if ( v17 )
          {
            if ( v16 )
            {
LABEL_55:
              if ( v17 )
              {
                sub_1A898D0(v46, v16);
                v24 = (_QWORD *)*v15;
                --a1[2];
                v42 = v24;
                sub_2208CA0(v15);
                if ( *((_BYTE *)v15 + 288) )
                {
                  v38 = *((unsigned int *)v15 + 70);
                  if ( (_DWORD)v38 )
                  {
                    v39 = (_QWORD *)v15[33];
                    v40 = &v39[2 * v38];
                    do
                    {
                      if ( *v39 != -8 && *v39 != -4 )
                      {
                        v41 = v39[1];
                        if ( v41 )
                          sub_161E7C0((__int64)(v39 + 1), v41);
                      }
                      v39 += 2;
                    }
                    while ( v40 != v39 );
                  }
                  j___libc_free_0(v15[33]);
                }
                v25 = *((_DWORD *)v15 + 62);
                if ( v25 )
                {
                  v26 = (_QWORD *)v15[29];
                  v49 = 2;
                  v27 = (unsigned __int64)v25 << 6;
                  v50 = 0;
                  v28 = -8;
                  v51 = -8;
                  v29 = (_QWORD *)((char *)v26 + v27);
                  v53 = &unk_49E6B50;
                  v52 = 0;
                  v54 = 2;
                  v55 = 0;
                  v56 = -16;
                  v57 = 0;
                  while ( 1 )
                  {
                    v30 = v26[3];
                    if ( v28 != v30 )
                    {
                      v28 = v56;
                      if ( v30 != v56 )
                      {
                        v31 = v26[7];
                        if ( v31 != 0 && v31 != -8 && v31 != -16 )
                        {
                          v47 = v26;
                          sub_1649B30(v26 + 5);
                          v26 = v47;
                          v30 = v47[3];
                        }
                        v28 = v30;
                      }
                    }
                    *v26 = &unk_49EE2B0;
                    if ( v28 != 0 && v28 != -8 && v28 != -16 )
                    {
                      v48 = v26;
                      sub_1649B30(v26 + 1);
                      v26 = v48;
                    }
                    v26 += 8;
                    if ( v29 == v26 )
                      break;
                    v28 = v51;
                  }
                  v53 = &unk_49EE2B0;
                  if ( v56 != -8 && v56 != 0 && v56 != -16 )
                    sub_1649B30(&v54);
                  if ( v51 != 0 && v51 != -8 && v51 != -16 )
                    sub_1649B30(&v49);
                }
                j___libc_free_0(v15[29]);
                v32 = v15[18];
                if ( (_QWORD *)v32 != v15 + 20 )
                  _libc_free(v32);
                v33 = v15[4];
                if ( v33 != v15[3] )
                  _libc_free(v33);
                j_j___libc_free_0(v15, 304);
                v15 = v42;
LABEL_40:
                if ( v15 == a1 )
                  return;
                continue;
              }
              goto LABEL_47;
            }
          }
          else
          {
            v18 = (_QWORD *)v15[4];
            if ( v18 == (_QWORD *)v15[3] )
              v19 = *((unsigned int *)v15 + 11);
            else
              v19 = *((unsigned int *)v15 + 10);
            v20 = &v18[v19];
            if ( v18 == v20 )
              goto LABEL_47;
            while ( 1 )
            {
              v21 = *v18;
              v22 = v18;
              if ( *v18 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v20 == ++v18 )
                goto LABEL_47;
            }
            if ( v20 == v18 )
              goto LABEL_47;
            if ( *(_BYTE *)(v21 + 16) == 55 )
              goto LABEL_83;
            while ( 1 )
            {
              v23 = v22 + 1;
              if ( v22 + 1 == v20 )
                break;
              while ( 1 )
              {
                v21 = *v23;
                v22 = v23;
                if ( *v23 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v20 == ++v23 )
                  goto LABEL_53;
              }
              if ( v20 == v23 )
                break;
              if ( *(_BYTE *)(v21 + 16) == 55 )
              {
LABEL_83:
                v17 = sub_3860560(*(_QWORD *)(v21 + 40), a1[7], a1[9]);
                if ( !v17 )
                {
                  v15 = (_QWORD *)*v15;
                  v16 = 0;
                  goto LABEL_40;
                }
              }
            }
LABEL_53:
            if ( (v17 & (v16 == 0)) == 0 )
            {
              if ( v16 )
                goto LABEL_55;
LABEL_47:
              v15 = (_QWORD *)*v15;
              v16 = 0;
              goto LABEL_40;
            }
          }
          break;
        }
        v15 = (_QWORD *)*v15;
        v16 = v46;
        goto LABEL_40;
      }
    }
  }
}
