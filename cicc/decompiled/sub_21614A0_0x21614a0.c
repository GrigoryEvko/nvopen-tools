// Function: sub_21614A0
// Address: 0x21614a0
//
__int64 __fastcall sub_21614A0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rcx
  bool v14; // zf
  bool v15; // si
  bool v16; // al
  __int64 v17; // rcx
  bool v18; // al
  __int64 v19; // r12
  __int64 v20; // rax
  unsigned int v21; // esi
  _QWORD *v22; // r13
  int v23; // eax
  int v24; // edx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdi
  unsigned int v28; // ecx
  _QWORD *v29; // rdx
  __int64 v30; // r9
  int v31; // edx
  __int64 v32; // rdi
  _QWORD *v33; // r8
  int v34; // r9d
  unsigned int v35; // eax
  __int64 v36; // rsi
  int v37; // r9d
  int v38; // r10d
  int v39; // eax
  int v40; // eax
  int v41; // eax
  __int64 v42; // rdi
  int v43; // r9d
  unsigned int v44; // edx
  __int64 v45; // rsi
  unsigned __int64 v46[2]; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v47; // [rsp+28h] [rbp-D8h]
  __int64 v48; // [rsp+30h] [rbp-D0h]
  void *v49; // [rsp+40h] [rbp-C0h]
  _QWORD v50[2]; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v51; // [rsp+58h] [rbp-A8h]
  __int64 v52; // [rsp+60h] [rbp-A0h]
  void *v53; // [rsp+70h] [rbp-90h]
  _QWORD v54[2]; // [rsp+78h] [rbp-88h] BYREF
  __int64 v55; // [rsp+88h] [rbp-78h]
  __int64 v56; // [rsp+90h] [rbp-70h]
  void *v57; // [rsp+A0h] [rbp-60h]
  unsigned __int64 v58; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v59; // [rsp+B0h] [rbp-50h]
  __int64 v60; // [rsp+B8h] [rbp-48h]
  __int64 v61; // [rsp+C0h] [rbp-40h]
  __int64 v62; // [rsp+C8h] [rbp-38h]

  v3 = a1[1];
  v46[1] = 0;
  v46[0] = v3 & 6;
  v47 = a1[3];
  result = v47;
  if ( v47 != 0 && v47 != -8 && v47 != -16 )
  {
    sub_1649AC0(v46, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v47;
  }
  v5 = a1[4];
  v48 = v5;
  v6 = *(unsigned int *)(v5 + 24);
  if ( !(_DWORD)v6 )
    goto LABEL_5;
  v7 = *(_QWORD *)(v5 + 8);
  v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v9 = (_QWORD *)(v7 + 48LL * v8);
  v10 = v9[3];
  if ( v10 == result )
  {
LABEL_10:
    if ( v9 == (_QWORD *)(v7 + 48 * v6) )
      goto LABEL_5;
    v11 = v9[5];
    v58 = 2;
    v59 = 0;
    v60 = -16;
    v57 = &unk_4A01B30;
    v61 = 0;
    v12 = v9[3];
    if ( v12 == -16 )
    {
      v9[4] = 0;
      goto LABEL_20;
    }
    if ( !v12 || v12 == -8 )
    {
      v9[3] = -16;
    }
    else
    {
      sub_1649B30(v9 + 1);
      v13 = v60;
      v14 = v60 == -8;
      v9[3] = v60;
      v15 = v13 != 0;
      v16 = v13 != -16;
      if ( v13 == 0 || v14 || v13 == -16 )
      {
        v17 = v61;
        v18 = v15 && !v14 && v16;
LABEL_18:
        v9[4] = v17;
        v57 = &unk_49EE2B0;
        if ( v18 )
          sub_1649B30(&v58);
LABEL_20:
        --*(_DWORD *)(v5 + 16);
        ++*(_DWORD *)(v5 + 20);
        v19 = v48;
        v50[0] = 2;
        v51 = a2;
        v50[1] = 0;
        if ( a2 == -8 || a2 == 0 || a2 == -16 )
        {
          v52 = v48;
          v49 = &unk_4A01B30;
          v20 = v48;
          v58 = 2;
          v59 = 0;
          v60 = a2;
        }
        else
        {
          sub_164C220((__int64)v50);
          v49 = &unk_4A01B30;
          v52 = v19;
          v59 = 0;
          v58 = v50[0] & 6;
          v60 = v51;
          if ( v51 == -8 || v51 == 0 || v51 == -16 )
          {
            v20 = v19;
          }
          else
          {
            sub_1649AC0(&v58, v50[0] & 0xFFFFFFFFFFFFFFF8LL);
            v20 = v52;
          }
        }
        v61 = v20;
        v57 = &unk_4A01B30;
        v62 = v11;
        v21 = *(_DWORD *)(v19 + 24);
        if ( v21 )
        {
          v26 = v60;
          v27 = *(_QWORD *)(v19 + 8);
          v28 = (v21 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v29 = (_QWORD *)(v27 + 48LL * v28);
          v30 = v29[3];
          if ( v30 == v60 )
          {
LABEL_45:
            v57 = &unk_49EE2B0;
            if ( v26 != 0 && v26 != -8 && v26 != -16 )
              sub_1649B30(&v58);
            v49 = &unk_49EE2B0;
            if ( v51 != 0 && v51 != -8 && v51 != -16 )
              sub_1649B30(v50);
            result = v47;
            goto LABEL_5;
          }
          v38 = 1;
          v22 = 0;
          while ( v30 != -8 )
          {
            if ( v30 != -16 || v22 )
              v29 = v22;
            v28 = (v21 - 1) & (v38 + v28);
            v30 = *(_QWORD *)(v27 + 48LL * v28 + 24);
            if ( v60 == v30 )
              goto LABEL_45;
            ++v38;
            v22 = v29;
            v29 = (_QWORD *)(v27 + 48LL * v28);
          }
          v39 = *(_DWORD *)(v19 + 16);
          if ( !v22 )
            v22 = v29;
          ++*(_QWORD *)v19;
          v24 = v39 + 1;
          if ( 4 * (v39 + 1) < 3 * v21 )
          {
            if ( v21 - *(_DWORD *)(v19 + 20) - v24 > v21 >> 3 )
            {
LABEL_29:
              *(_DWORD *)(v19 + 16) = v24;
              v55 = -8;
              v56 = 0;
              v14 = v22[3] == -8;
              v54[0] = 2;
              v54[1] = 0;
              if ( !v14 )
              {
                --*(_DWORD *)(v19 + 20);
                v53 = &unk_49EE2B0;
                if ( v55 != -8 && v55 != 0 && v55 != -16 )
                  sub_1649B30(v54);
              }
              v25 = v22[3];
              v26 = v60;
              if ( v25 != v60 )
              {
                if ( v25 != 0 && v25 != -8 && v25 != -16 )
                {
                  sub_1649B30(v22 + 1);
                  v26 = v60;
                }
                v22[3] = v26;
                if ( v26 != 0 && v26 != -8 && v26 != -16 )
                  sub_1649AC0(v22 + 1, v58 & 0xFFFFFFFFFFFFFFF8LL);
                v26 = v60;
              }
              v22[4] = v61;
              v22[5] = v62;
              goto LABEL_45;
            }
            v22 = 0;
            sub_215DD00(v19, v21);
            v40 = *(_DWORD *)(v19 + 24);
            if ( v40 )
            {
              v41 = v40 - 1;
              v42 = *(_QWORD *)(v19 + 8);
              v33 = 0;
              v43 = 1;
              v44 = v41 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
              v22 = (_QWORD *)(v42 + 48LL * v44);
              v45 = v22[3];
              if ( v45 != v60 )
              {
                while ( v45 != -8 )
                {
                  if ( v45 == -16 && !v33 )
                    v33 = v22;
                  v44 = v41 & (v43 + v44);
                  v22 = (_QWORD *)(v42 + 48LL * v44);
                  v45 = v22[3];
                  if ( v60 == v45 )
                    goto LABEL_28;
                  ++v43;
                }
                goto LABEL_56;
              }
            }
LABEL_28:
            v24 = *(_DWORD *)(v19 + 16) + 1;
            goto LABEL_29;
          }
        }
        else
        {
          ++*(_QWORD *)v19;
        }
        v22 = 0;
        sub_215DD00(v19, 2 * v21);
        v23 = *(_DWORD *)(v19 + 24);
        if ( v23 )
        {
          v31 = v23 - 1;
          v32 = *(_QWORD *)(v19 + 8);
          v33 = 0;
          v34 = 1;
          v35 = (v23 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v22 = (_QWORD *)(v32 + 48LL * v35);
          v36 = v22[3];
          if ( v36 != v60 )
          {
            while ( v36 != -8 )
            {
              if ( v36 == -16 && !v33 )
                v33 = v22;
              v35 = v31 & (v34 + v35);
              v22 = (_QWORD *)(v32 + 48LL * v35);
              v36 = v22[3];
              if ( v60 == v36 )
                goto LABEL_28;
              ++v34;
            }
LABEL_56:
            if ( v33 )
              v22 = v33;
            goto LABEL_28;
          }
        }
        goto LABEL_28;
      }
      sub_1649AC0(v9 + 1, v58 & 0xFFFFFFFFFFFFFFF8LL);
    }
    v17 = v61;
    v18 = v60 != 0 && v60 != -16 && v60 != -8;
    goto LABEL_18;
  }
  v37 = 1;
  while ( v10 != -8 )
  {
    v8 = (v6 - 1) & (v37 + v8);
    v9 = (_QWORD *)(v7 + 48LL * v8);
    v10 = v9[3];
    if ( v10 == result )
      goto LABEL_10;
    ++v37;
  }
LABEL_5:
  if ( result != 0 && result != -8 && result != -16 )
    return sub_1649B30(v46);
  return result;
}
