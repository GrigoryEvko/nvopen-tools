// Function: sub_1B56760
// Address: 0x1b56760
//
__int64 __fastcall sub_1B56760(__int64 *a1, __int64 a2)
{
  _QWORD *v3; // r12
  __int64 **v4; // rax
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  int v7; // esi
  unsigned int v8; // edi
  __int64 *v9; // r9
  __int64 v10; // r10
  __int64 **v11; // rdi
  __int64 v12; // r13
  __int64 v13; // r14
  _QWORD *v14; // rax
  __int64 v15; // r13
  int v16; // r11d
  __int64 **v17; // r10
  unsigned int v18; // edx
  __int64 **v19; // rdi
  __int64 *v20; // rcx
  __int64 *v21; // rax
  int v22; // edi
  unsigned int v23; // edx
  __int64 *v24; // rsi
  __int64 **v25; // rax
  __int64 **v26; // r9
  int v27; // r11d
  unsigned int v28; // edx
  unsigned int v29; // r13d
  __int64 v30; // rdi
  __int64 *v31; // rbx
  __int64 *v32; // r12
  __int64 v33; // rdi
  __int64 v35; // rdx
  __int64 *v36; // r8
  int v37; // eax
  int v38; // r13d
  int v39; // r11d
  __int64 **v40; // r9
  __int64 v41[2]; // [rsp+8h] [rbp-B8h] BYREF
  __int64 *v42; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v43; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v44; // [rsp+28h] [rbp-98h]
  __int64 v45; // [rsp+30h] [rbp-90h]
  __int64 v46; // [rsp+38h] [rbp-88h]
  __int64 v47; // [rsp+40h] [rbp-80h] BYREF
  __int64 v48; // [rsp+48h] [rbp-78h]
  __int64 **v49; // [rsp+50h] [rbp-70h]
  __int64 **v50; // [rsp+58h] [rbp-68h]
  _QWORD *v51; // [rsp+60h] [rbp-60h]
  unsigned __int64 v52; // [rsp+68h] [rbp-58h]
  __int64 **v53; // [rsp+70h] [rbp-50h]
  __int64 **v54; // [rsp+78h] [rbp-48h]
  _QWORD *v55; // [rsp+80h] [rbp-40h]
  _QWORD *v56; // [rsp+88h] [rbp-38h]

  v41[0] = a2;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v48 = 8;
  v47 = sub_22077B0(64);
  v3 = (_QWORD *)(v47 + 24);
  v4 = (__int64 **)sub_22077B0(512);
  v5 = a2;
  v52 = v47 + 24;
  v6 = v4 + 64;
  *(_QWORD *)(v47 + 24) = v4;
  v50 = v4;
  v51 = v4 + 64;
  v56 = v3;
  v54 = v4;
  v55 = v4 + 64;
  v49 = v4;
  v53 = v4;
  if ( (__int64 *)a2 != a1 )
  {
    v7 = v46;
    if ( (_DWORD)v46 )
    {
      v8 = (v46 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v9 = (__int64 *)(v44 + 8LL * v8);
      v10 = *v9;
      if ( v5 == *v9 )
      {
LABEL_4:
        if ( v4 == v6 - 1 )
        {
          sub_1B4EEC0(&v47, v41);
          v11 = v53;
        }
        else
        {
          if ( v4 )
          {
            *v4 = (__int64 *)v41[0];
            v4 = v53;
          }
          v11 = v4 + 1;
          v53 = v4 + 1;
        }
        while ( 1 )
        {
          if ( v49 == v11 )
            goto LABEL_56;
          if ( v54 == v11 )
          {
            v12 = *(_QWORD *)(*(v56 - 1) + 504LL);
            j_j___libc_free_0(v11, 512);
            v35 = *--v56 + 512LL;
            v54 = (__int64 **)*v56;
            v55 = (_QWORD *)v35;
            v53 = v54 + 63;
          }
          else
          {
            v12 = (__int64)*(v11 - 1);
            v53 = v11 - 1;
          }
          v13 = *(_QWORD *)(v12 + 8);
          if ( !v13 )
          {
LABEL_47:
            v29 = 0;
            goto LABEL_48;
          }
          while ( 1 )
          {
            v14 = sub_1648700(v13);
            if ( (unsigned __int8)(*((_BYTE *)v14 + 16) - 25) <= 9u )
              break;
            v13 = *(_QWORD *)(v13 + 8);
            if ( !v13 )
              goto LABEL_47;
          }
          v15 = 0;
          while ( 1 )
          {
            v21 = (__int64 *)v14[5];
            ++v15;
            v42 = v21;
            if ( a1 == v21 )
              goto LABEL_15;
            if ( !(_DWORD)v46 )
            {
              ++v43;
LABEL_20:
              sub_163D380((__int64)&v43, 2 * v46);
              if ( !(_DWORD)v46 )
                goto LABEL_92;
              v21 = v42;
              v22 = v45 + 1;
              v23 = (v46 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
              v17 = (__int64 **)(v44 + 8LL * v23);
              v24 = *v17;
              if ( *v17 != v42 )
              {
                v39 = 1;
                v40 = 0;
                while ( v24 != (__int64 *)-8LL )
                {
                  if ( !v40 && v24 == (__int64 *)-16LL )
                    v40 = v17;
                  v23 = (v46 - 1) & (v39 + v23);
                  v17 = (__int64 **)(v44 + 8LL * v23);
                  v24 = *v17;
                  if ( v42 == *v17 )
                    goto LABEL_22;
                  ++v39;
                }
                if ( v40 )
                  v17 = v40;
              }
              goto LABEL_22;
            }
            v16 = 1;
            v17 = 0;
            v18 = (v46 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v19 = (__int64 **)(v44 + 8LL * v18);
            v20 = *v19;
            if ( v21 != *v19 )
              break;
LABEL_15:
            while ( 2 )
            {
              v13 = *(_QWORD *)(v13 + 8);
              if ( !v13 )
                goto LABEL_29;
LABEL_16:
              v14 = sub_1648700(v13);
              if ( (unsigned __int8)(*((_BYTE *)v14 + 16) - 25) > 9u )
                continue;
              break;
            }
          }
          while ( v20 != (__int64 *)-8LL )
          {
            if ( v17 || v20 != (__int64 *)-16LL )
              v19 = v17;
            v18 = (v46 - 1) & (v16 + v18);
            v20 = *(__int64 **)(v44 + 8LL * v18);
            if ( v21 == v20 )
              goto LABEL_15;
            ++v16;
            v17 = v19;
            v19 = (__int64 **)(v44 + 8LL * v18);
          }
          if ( !v17 )
            v17 = v19;
          ++v43;
          v22 = v45 + 1;
          if ( 4 * ((int)v45 + 1) >= (unsigned int)(3 * v46) )
            goto LABEL_20;
          if ( (int)v46 - HIDWORD(v45) - v22 <= (unsigned int)v46 >> 3 )
          {
            sub_163D380((__int64)&v43, v46);
            if ( !(_DWORD)v46 )
            {
LABEL_92:
              LODWORD(v45) = v45 + 1;
              BUG();
            }
            v26 = 0;
            v27 = 1;
            v22 = v45 + 1;
            v28 = (v46 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
            v17 = (__int64 **)(v44 + 8LL * v28);
            v21 = *v17;
            if ( v42 != *v17 )
            {
              while ( v21 != (__int64 *)-8LL )
              {
                if ( !v26 && v21 == (__int64 *)-16LL )
                  v26 = v17;
                v28 = (v46 - 1) & (v27 + v28);
                v17 = (__int64 **)(v44 + 8LL * v28);
                v21 = *v17;
                if ( v42 == *v17 )
                  goto LABEL_22;
                ++v27;
              }
              v21 = v42;
              if ( v26 )
                v17 = v26;
            }
          }
LABEL_22:
          LODWORD(v45) = v22;
          if ( *v17 != (__int64 *)-8LL )
            --HIDWORD(v45);
          *v17 = v21;
          if ( (unsigned int)v45 > 0x64 )
            goto LABEL_47;
          v25 = v53;
          if ( v53 == v55 - 1 )
          {
            sub_1B4EEC0(&v47, &v42);
            goto LABEL_15;
          }
          if ( v53 )
          {
            *v53 = v42;
            v25 = v53;
          }
          v53 = v25 + 1;
          v13 = *(_QWORD *)(v13 + 8);
          if ( v13 )
            goto LABEL_16;
LABEL_29:
          if ( !v15 )
            goto LABEL_47;
          v11 = v53;
        }
      }
      v38 = 1;
      v36 = 0;
      while ( v10 != -8 )
      {
        if ( !v36 && v10 == -16 )
          v36 = v9;
        v8 = (v46 - 1) & (v38 + v8);
        v9 = (__int64 *)(v44 + 8LL * v8);
        v10 = *v9;
        if ( v5 == *v9 )
          goto LABEL_4;
        ++v38;
      }
      if ( !v36 )
        v36 = v9;
      ++v43;
      v37 = v45 + 1;
      if ( 4 * ((int)v45 + 1) < (unsigned int)(3 * v46) )
      {
        if ( (int)v46 - HIDWORD(v45) - v37 > (unsigned int)v46 >> 3 )
          goto LABEL_61;
        goto LABEL_60;
      }
    }
    else
    {
      ++v43;
    }
    v7 = 2 * v46;
LABEL_60:
    sub_163D380((__int64)&v43, v7);
    sub_1B50600((__int64)&v43, v41, &v42);
    v36 = v42;
    v5 = v41[0];
    v37 = v45 + 1;
LABEL_61:
    LODWORD(v45) = v37;
    if ( *v36 != -8 )
      --HIDWORD(v45);
    *v36 = v5;
    v4 = v53;
    v6 = v55;
    goto LABEL_4;
  }
LABEL_56:
  v29 = 1;
LABEL_48:
  v30 = v47;
  if ( v47 )
  {
    v31 = (__int64 *)v52;
    v32 = v56 + 1;
    if ( (unsigned __int64)(v56 + 1) > v52 )
    {
      do
      {
        v33 = *v31++;
        j_j___libc_free_0(v33, 512);
      }
      while ( v32 > v31 );
      v30 = v47;
    }
    j_j___libc_free_0(v30, 8 * v48);
  }
  j___libc_free_0(v44);
  return v29;
}
