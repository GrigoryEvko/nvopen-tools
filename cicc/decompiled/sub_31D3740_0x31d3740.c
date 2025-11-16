// Function: sub_31D3740
// Address: 0x31d3740
//
__int64 __fastcall sub_31D3740(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rcx
  bool v16; // al
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned int v19; // esi
  _QWORD *v20; // r12
  int v21; // eax
  int v22; // edx
  __int64 v23; // rdx
  __int64 v24; // rax
  int v25; // r9d
  __int64 v26; // rdi
  unsigned int v27; // ecx
  _QWORD *v28; // rdx
  __int64 v29; // r9
  int v30; // edx
  __int64 v31; // rdi
  _QWORD *v32; // r8
  int v33; // r9d
  unsigned int v34; // eax
  __int64 v35; // rsi
  int v36; // r10d
  int v37; // eax
  int v38; // eax
  int v39; // eax
  __int64 v40; // rdi
  int v41; // r9d
  unsigned int v42; // edx
  __int64 v43; // rsi
  unsigned __int64 v44[2]; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v45; // [rsp+28h] [rbp-D8h]
  __int64 v46; // [rsp+30h] [rbp-D0h]
  void *v47; // [rsp+40h] [rbp-C0h]
  _QWORD v48[2]; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v49; // [rsp+58h] [rbp-A8h]
  __int64 v50; // [rsp+60h] [rbp-A0h]
  void *v51; // [rsp+70h] [rbp-90h]
  _QWORD v52[2]; // [rsp+78h] [rbp-88h] BYREF
  __int64 v53; // [rsp+88h] [rbp-78h]
  __int64 v54; // [rsp+90h] [rbp-70h]
  void *v55; // [rsp+A0h] [rbp-60h]
  unsigned __int64 v56; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v57; // [rsp+B0h] [rbp-50h]
  __int64 v58; // [rsp+B8h] [rbp-48h]
  __int64 v59; // [rsp+C0h] [rbp-40h]
  __int64 v60; // [rsp+C8h] [rbp-38h]

  v3 = a1[1];
  v44[1] = 0;
  v44[0] = v3 & 6;
  v45 = a1[3];
  result = v45;
  if ( v45 != -4096 && v45 != 0 && v45 != -8192 )
  {
    sub_BD6050(v44, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v45;
  }
  v5 = a1[4];
  v46 = v5;
  v6 = *(unsigned int *)(v5 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v9 = (_QWORD *)(v7 + 48LL * v8);
    v10 = v9[3];
    if ( v10 == result )
    {
LABEL_6:
      if ( v9 == (_QWORD *)(v7 + 48 * v6) )
        goto LABEL_39;
      v11 = v9[5];
      v56 = 2;
      v57 = 0;
      v58 = -8192;
      v55 = &unk_4A34DD0;
      v59 = 0;
      v12 = v9[3];
      if ( v12 == -8192 )
      {
        v9[4] = 0;
      }
      else
      {
        if ( !v12 || v12 == -4096 )
        {
          v9[3] = -8192;
          v15 = v59;
          v16 = v58 != -4096 && v58 != -8192 && v58 != 0;
        }
        else
        {
          sub_BD60C0(v9 + 1);
          v13 = v58;
          v14 = v58 == 0;
          v9[3] = v58;
          if ( v13 == -4096 || v14 || v13 == -8192 )
          {
            v9[4] = v59;
            goto LABEL_15;
          }
          sub_BD6050(v9 + 1, v56 & 0xFFFFFFFFFFFFFFF8LL);
          v15 = v59;
          v16 = v58 != -8192 && v58 != 0 && v58 != -4096;
        }
        v9[4] = v15;
        v55 = &unk_49DB368;
        if ( v16 )
          sub_BD60C0(&v56);
      }
LABEL_15:
      --*(_DWORD *)(v5 + 16);
      ++*(_DWORD *)(v5 + 20);
      v17 = v46;
      v48[0] = 2;
      v49 = a2;
      v48[1] = 0;
      if ( a2 == 0 || a2 == -4096 || a2 == -8192 )
      {
        v50 = v46;
        v47 = &unk_4A34DD0;
        v18 = v46;
        v56 = 2;
        v57 = 0;
        v58 = a2;
      }
      else
      {
        sub_BD73F0((__int64)v48);
        v47 = &unk_4A34DD0;
        v50 = v17;
        v57 = 0;
        v56 = v48[0] & 6;
        v58 = v49;
        if ( v49 == 0 || v49 == -4096 || v49 == -8192 )
        {
          v18 = v17;
        }
        else
        {
          sub_BD6050(&v56, v48[0] & 0xFFFFFFFFFFFFFFF8LL);
          v18 = v50;
        }
      }
      v59 = v18;
      v55 = &unk_4A34DD0;
      v60 = v11;
      v19 = *(_DWORD *)(v17 + 24);
      if ( v19 )
      {
        v24 = v58;
        v26 = *(_QWORD *)(v17 + 8);
        v27 = (v19 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v28 = (_QWORD *)(v26 + 48LL * v27);
        v29 = v28[3];
        if ( v58 == v29 )
        {
LABEL_47:
          v55 = &unk_49DB368;
          if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
            sub_BD60C0(&v56);
          v47 = &unk_49DB368;
          if ( v49 != 0 && v49 != -4096 && v49 != -8192 )
            sub_BD60C0(v48);
          result = v45;
          goto LABEL_39;
        }
        v36 = 1;
        v20 = 0;
        while ( v29 != -4096 )
        {
          if ( v20 || v29 != -8192 )
            v28 = v20;
          v27 = (v19 - 1) & (v36 + v27);
          v29 = *(_QWORD *)(v26 + 48LL * v27 + 24);
          if ( v58 == v29 )
            goto LABEL_47;
          ++v36;
          v20 = v28;
          v28 = (_QWORD *)(v26 + 48LL * v27);
        }
        v37 = *(_DWORD *)(v17 + 16);
        if ( !v20 )
          v20 = v28;
        ++*(_QWORD *)v17;
        v22 = v37 + 1;
        if ( 4 * (v37 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(v17 + 20) - v22 > v19 >> 3 )
          {
LABEL_24:
            *(_DWORD *)(v17 + 16) = v22;
            v53 = -4096;
            v54 = 0;
            v14 = v20[3] == -4096;
            v52[0] = 2;
            v52[1] = 0;
            if ( !v14 )
            {
              --*(_DWORD *)(v17 + 20);
              v51 = &unk_49DB368;
              if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
                sub_BD60C0(v52);
            }
            v23 = v20[3];
            v24 = v58;
            if ( v23 != v58 )
            {
              if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
              {
                sub_BD60C0(v20 + 1);
                v24 = v58;
              }
              v20[3] = v24;
              if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
                sub_BD6050(v20 + 1, v56 & 0xFFFFFFFFFFFFFFF8LL);
              v24 = v58;
            }
            v20[4] = v59;
            v20[5] = v60;
            goto LABEL_47;
          }
          v20 = 0;
          sub_31CF910(v17, v19);
          v38 = *(_DWORD *)(v17 + 24);
          if ( v38 )
          {
            v39 = v38 - 1;
            v40 = *(_QWORD *)(v17 + 8);
            v32 = 0;
            v41 = 1;
            v42 = v39 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
            v20 = (_QWORD *)(v40 + 48LL * v42);
            v43 = v20[3];
            if ( v43 != v58 )
            {
              while ( v43 != -4096 )
              {
                if ( v43 == -8192 && !v32 )
                  v32 = v20;
                v42 = v39 & (v41 + v42);
                v20 = (_QWORD *)(v40 + 48LL * v42);
                v43 = v20[3];
                if ( v58 == v43 )
                  goto LABEL_23;
                ++v41;
              }
              goto LABEL_57;
            }
          }
LABEL_23:
          v22 = *(_DWORD *)(v17 + 16) + 1;
          goto LABEL_24;
        }
      }
      else
      {
        ++*(_QWORD *)v17;
      }
      v20 = 0;
      sub_31CF910(v17, 2 * v19);
      v21 = *(_DWORD *)(v17 + 24);
      if ( v21 )
      {
        v30 = v21 - 1;
        v31 = *(_QWORD *)(v17 + 8);
        v32 = 0;
        v33 = 1;
        v34 = (v21 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v20 = (_QWORD *)(v31 + 48LL * v34);
        v35 = v20[3];
        if ( v58 != v35 )
        {
          while ( v35 != -4096 )
          {
            if ( v35 == -8192 && !v32 )
              v32 = v20;
            v34 = v30 & (v33 + v34);
            v20 = (_QWORD *)(v31 + 48LL * v34);
            v35 = v20[3];
            if ( v58 == v35 )
              goto LABEL_23;
            ++v33;
          }
LABEL_57:
          if ( v32 )
            v20 = v32;
          goto LABEL_23;
        }
      }
      goto LABEL_23;
    }
    v25 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v25 + v8);
      v9 = (_QWORD *)(v7 + 48LL * v8);
      v10 = v9[3];
      if ( v10 == result )
        goto LABEL_6;
      ++v25;
    }
  }
LABEL_39:
  if ( result != 0 && result != -4096 && result != -8192 )
    return sub_BD60C0(v44);
  return result;
}
