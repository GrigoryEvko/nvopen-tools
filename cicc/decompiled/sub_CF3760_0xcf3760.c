// Function: sub_CF3760
// Address: 0xcf3760
//
__int64 __fastcall sub_CF3760(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned int v19; // esi
  _QWORD *v20; // r12
  int v21; // eax
  int v22; // edx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // r9d
  __int64 v28; // r8
  unsigned int v29; // edx
  _QWORD *v30; // rax
  __int64 v31; // rdi
  int v32; // edx
  __int64 v33; // rdi
  _QWORD *v34; // r8
  int v35; // r9d
  unsigned int v36; // eax
  __int64 v37; // rsi
  int v38; // r10d
  int v39; // eax
  int v40; // eax
  int v41; // eax
  __int64 v42; // rdi
  int v43; // r9d
  unsigned int v44; // edx
  __int64 v45; // rsi
  unsigned __int64 v46[2]; // [rsp+0h] [rbp-140h] BYREF
  __int64 v47; // [rsp+10h] [rbp-130h]
  __int64 v48; // [rsp+20h] [rbp-120h]
  unsigned __int64 v49[2]; // [rsp+28h] [rbp-118h] BYREF
  __int64 v50; // [rsp+38h] [rbp-108h]
  void *v51; // [rsp+40h] [rbp-100h]
  unsigned __int64 v52[2]; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v53; // [rsp+58h] [rbp-E8h]
  __int64 v54; // [rsp+60h] [rbp-E0h]
  void *v55; // [rsp+70h] [rbp-D0h]
  _QWORD v56[2]; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v57; // [rsp+88h] [rbp-B8h]
  __int64 v58; // [rsp+90h] [rbp-B0h]
  void *v59; // [rsp+A0h] [rbp-A0h]
  _QWORD v60[2]; // [rsp+A8h] [rbp-98h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-88h]
  __int64 v62; // [rsp+C0h] [rbp-80h]
  void *v63; // [rsp+D0h] [rbp-70h]
  unsigned __int64 v64; // [rsp+D8h] [rbp-68h] BYREF
  __int64 v65; // [rsp+E0h] [rbp-60h]
  __int64 v66; // [rsp+E8h] [rbp-58h]
  __int64 v67; // [rsp+F0h] [rbp-50h]
  unsigned __int64 v68[2]; // [rsp+F8h] [rbp-48h] BYREF
  __int64 v69; // [rsp+108h] [rbp-38h]

  v3 = a1[1];
  v52[1] = 0;
  v52[0] = v3 & 6;
  v53 = a1[3];
  result = v53;
  if ( v53 != -4096 && v53 != 0 && v53 != -8192 )
  {
    sub_BD6050(v52, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v53;
  }
  v5 = a1[4];
  v54 = v5;
  v51 = &unk_49DD7B0;
  v6 = *(unsigned int *)(v5 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v9 = (_QWORD *)(v7 + ((unsigned __int64)v8 << 6));
    v10 = v9[3];
    if ( result == v10 )
    {
LABEL_6:
      if ( v9 == (_QWORD *)(v7 + (v6 << 6)) )
        goto LABEL_51;
      v46[0] = 6;
      v11 = v9[7];
      v46[1] = 0;
      v47 = v11;
      if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
      {
        sub_BD6050(v46, v9[5] & 0xFFFFFFFFFFFFFFF8LL);
        v12 = v9[7];
        v5 = v54;
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD60C0(v9 + 5);
      }
      v64 = 2;
      v65 = 0;
      v66 = -8192;
      v63 = &unk_49DD7B0;
      v67 = 0;
      v13 = v9[3];
      if ( v13 == -8192 )
      {
        v9[4] = 0;
      }
      else
      {
        if ( v13 == -4096 || !v13 )
        {
          v9[3] = -8192;
        }
        else
        {
          sub_BD60C0(v9 + 1);
          v14 = v66;
          v15 = v66 == -4096;
          v9[3] = v66;
          if ( v14 == 0 || v15 || v14 == -8192 )
          {
            v9[4] = v67;
            goto LABEL_20;
          }
          sub_BD6050(v9 + 1, v64 & 0xFFFFFFFFFFFFFFF8LL);
        }
        v16 = v66;
        v15 = v66 == 0;
        v9[4] = v67;
        v63 = &unk_49DB368;
        if ( v16 != -8192 && v16 != -4096 && !v15 )
          sub_BD60C0(&v64);
      }
LABEL_20:
      --*(_DWORD *)(v5 + 16);
      ++*(_DWORD *)(v5 + 20);
      v48 = a2;
      v17 = v54;
      v50 = v47;
      v49[0] = 6;
      v49[1] = 0;
      if ( v47 != -4096 && v47 != 0 && v47 != -8192 )
      {
        sub_BD6050(v49, v46[0] & 0xFFFFFFFFFFFFFFF8LL);
        a2 = v48;
      }
      v57 = a2;
      v56[0] = 2;
      v56[1] = 0;
      if ( a2 == 0 || a2 == -4096 || a2 == -8192 )
      {
        v58 = v17;
        v55 = &unk_49DD7B0;
        v18 = v17;
        v64 = 2;
        v65 = 0;
        v66 = a2;
      }
      else
      {
        sub_BD73F0((__int64)v56);
        v55 = &unk_49DD7B0;
        v58 = v17;
        v65 = 0;
        v64 = v56[0] & 6;
        v66 = v57;
        if ( v57 == -4096 || v57 == 0 || v57 == -8192 )
        {
          v18 = v17;
        }
        else
        {
          sub_BD6050(&v64, v56[0] & 0xFFFFFFFFFFFFFFF8LL);
          v18 = v58;
        }
      }
      v67 = v18;
      v63 = &unk_49DD7B0;
      v68[0] = 6;
      v68[1] = 0;
      v69 = v50;
      if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
        sub_BD6050(v68, v49[0] & 0xFFFFFFFFFFFFFFF8LL);
      v19 = *(_DWORD *)(v17 + 24);
      if ( v19 )
      {
        v28 = *(_QWORD *)(v17 + 8);
        v29 = (v19 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v30 = (_QWORD *)(v28 + ((unsigned __int64)v29 << 6));
        v31 = v30[3];
        if ( v31 == v66 )
        {
LABEL_59:
          if ( v69 != 0 && v69 != -4096 && v69 != -8192 )
            sub_BD60C0(v68);
          v63 = &unk_49DB368;
          if ( v66 != 0 && v66 != -4096 && v66 != -8192 )
            sub_BD60C0(&v64);
          v55 = &unk_49DB368;
          if ( v57 != 0 && v57 != -4096 && v57 != -8192 )
            sub_BD60C0(v56);
          if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
            sub_BD60C0(v49);
          if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
            sub_BD60C0(v46);
          result = v53;
          goto LABEL_51;
        }
        v38 = 1;
        v20 = 0;
        while ( v31 != -4096 )
        {
          if ( v20 || v31 != -8192 )
            v30 = v20;
          v29 = (v19 - 1) & (v38 + v29);
          v31 = *(_QWORD *)(v28 + ((unsigned __int64)v29 << 6) + 24);
          if ( v66 == v31 )
            goto LABEL_59;
          ++v38;
          v20 = v30;
          v30 = (_QWORD *)(v28 + ((unsigned __int64)v29 << 6));
        }
        if ( !v20 )
          v20 = v30;
        v39 = *(_DWORD *)(v17 + 16);
        ++*(_QWORD *)v17;
        v22 = v39 + 1;
        if ( 4 * (v39 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(v17 + 20) - v22 > v19 >> 3 )
          {
LABEL_35:
            *(_DWORD *)(v17 + 16) = v22;
            v61 = -4096;
            v62 = 0;
            v15 = v20[3] == -4096;
            v60[0] = 2;
            v60[1] = 0;
            if ( !v15 )
            {
              --*(_DWORD *)(v17 + 20);
              v59 = &unk_49DB368;
              if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
                sub_BD60C0(v60);
            }
            v23 = v20[3];
            v24 = v66;
            if ( v23 != v66 )
            {
              if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
              {
                sub_BD60C0(v20 + 1);
                v24 = v66;
              }
              v20[3] = v24;
              if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
                sub_BD6050(v20 + 1, v64 & 0xFFFFFFFFFFFFFFF8LL);
            }
            v25 = v67;
            v20[5] = 6;
            v20[6] = 0;
            v20[4] = v25;
            v26 = v69;
            v15 = v69 == -4096;
            v20[7] = v69;
            if ( v26 != 0 && !v15 && v26 != -8192 )
              sub_BD6050(v20 + 5, v68[0] & 0xFFFFFFFFFFFFFFF8LL);
            goto LABEL_59;
          }
          v20 = 0;
          sub_CF32C0(v17, v19);
          v40 = *(_DWORD *)(v17 + 24);
          if ( v40 )
          {
            v41 = v40 - 1;
            v42 = *(_QWORD *)(v17 + 8);
            v34 = 0;
            v43 = 1;
            v44 = v41 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
            v20 = (_QWORD *)(v42 + ((unsigned __int64)v44 << 6));
            v45 = v20[3];
            if ( v66 != v45 )
            {
              while ( v45 != -4096 )
              {
                if ( !v34 && v45 == -8192 )
                  v34 = v20;
                v44 = v41 & (v43 + v44);
                v20 = (_QWORD *)(v42 + ((unsigned __int64)v44 << 6));
                v45 = v20[3];
                if ( v66 == v45 )
                  goto LABEL_34;
                ++v43;
              }
              goto LABEL_78;
            }
          }
LABEL_34:
          v22 = *(_DWORD *)(v17 + 16) + 1;
          goto LABEL_35;
        }
      }
      else
      {
        ++*(_QWORD *)v17;
      }
      v20 = 0;
      sub_CF32C0(v17, 2 * v19);
      v21 = *(_DWORD *)(v17 + 24);
      if ( v21 )
      {
        v32 = v21 - 1;
        v33 = *(_QWORD *)(v17 + 8);
        v34 = 0;
        v35 = 1;
        v36 = (v21 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v20 = (_QWORD *)(v33 + ((unsigned __int64)v36 << 6));
        v37 = v20[3];
        if ( v37 != v66 )
        {
          while ( v37 != -4096 )
          {
            if ( !v34 && v37 == -8192 )
              v34 = v20;
            v36 = v32 & (v35 + v36);
            v20 = (_QWORD *)(v33 + ((unsigned __int64)v36 << 6));
            v37 = v20[3];
            if ( v66 == v37 )
              goto LABEL_34;
            ++v35;
          }
LABEL_78:
          if ( v34 )
            v20 = v34;
          goto LABEL_34;
        }
      }
      goto LABEL_34;
    }
    v27 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v27 + v8);
      v9 = (_QWORD *)(v7 + ((unsigned __int64)v8 << 6));
      v10 = v9[3];
      if ( v10 == result )
        goto LABEL_6;
      ++v27;
    }
  }
LABEL_51:
  v51 = &unk_49DB368;
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0(v52);
  return result;
}
