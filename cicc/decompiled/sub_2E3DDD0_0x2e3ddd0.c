// Function: sub_2E3DDD0
// Address: 0x2e3ddd0
//
__int64 __fastcall sub_2E3DDD0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // eax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r9
  unsigned int v8; // ebx
  unsigned int *v10; // r12
  __int64 v11; // r15
  __int64 v12; // r8
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r14
  _DWORD *v15; // rbx
  _DWORD *v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned int *v20; // r14
  unsigned int *v21; // rbx
  __int64 result; // rax
  int v23; // edx
  unsigned int *v24; // rcx
  unsigned int v25; // edi
  int v26; // r8d
  unsigned int *v27; // rax
  int v28; // edx
  __int64 v29; // r14
  __int64 *v30; // rbx
  __int64 v31; // rax
  _DWORD *v32; // rdi
  __int64 v33; // r15
  __int64 v34; // rax
  _QWORD *v35; // r14
  __int64 v36; // rax
  unsigned int *v37; // rbx
  unsigned int *v38; // r14
  unsigned __int64 *v39; // r8
  unsigned int v40; // edi
  int v41; // esi
  unsigned int *v42; // rcx
  int v43; // esi
  unsigned int v44; // edi
  _QWORD *v45; // [rsp+20h] [rbp-D0h]
  int v46; // [rsp+28h] [rbp-C8h]
  unsigned int v47; // [rsp+2Ch] [rbp-C4h]
  unsigned __int64 v48; // [rsp+30h] [rbp-C0h]
  char v49; // [rsp+38h] [rbp-B8h]
  __int64 v50; // [rsp+40h] [rbp-B0h] BYREF
  _DWORD *v51; // [rsp+48h] [rbp-A8h]
  __int64 v52; // [rsp+50h] [rbp-A0h]
  __int64 v53; // [rsp+58h] [rbp-98h]
  unsigned __int64 v54[2]; // [rsp+60h] [rbp-90h] BYREF
  _BYTE v55[64]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v56; // [rsp+B0h] [rbp-40h]
  char v57; // [rsp+B8h] [rbp-38h]

  v3 = a2;
  v4 = *(_DWORD *)(a2 + 12);
  if ( v4 > 1 )
  {
    v57 = 0;
    v54[1] = 0x400000000LL;
    v54[0] = (unsigned __int64)v55;
    v56 = 0;
    v50 = 1;
    v51 = 0;
    v52 = 0;
    v53 = 0;
    v5 = (4 * v4 / 3 + 1) | ((unsigned __int64)(4 * v4 / 3 + 1) >> 1);
    v6 = (((v5 >> 2) | v5) >> 4) | (v5 >> 2) | v5;
    sub_A08C50((__int64)&v50, ((((v6 >> 8) | v6) >> 16) | (v6 >> 8) | v6) + 1);
    v46 = *(_DWORD *)(a2 + 12);
    if ( !v46 )
    {
      v14 = 1;
      goto LABEL_13;
    }
    v49 = 0;
    v8 = 0;
    v45 = (_QWORD *)(a1 + 32);
    v48 = 0;
    v47 = 0;
    v46 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = (unsigned int *)(*(_QWORD *)(a2 + 96) + 4LL * v8);
        v11 = *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8LL * *v10);
        sub_FDE240(v45, *v10);
        if ( *(_BYTE *)(v11 + 176) )
          break;
        if ( !(_DWORD)v53 )
        {
          ++v50;
          goto LABEL_66;
        }
        v7 = (__int64)v51;
        v23 = (v53 - 1) & v47;
        v24 = &v51[v23];
        v25 = *v24;
        if ( *v24 != v8 )
        {
          v26 = 1;
          v27 = 0;
          while ( v25 != -1 )
          {
            if ( v25 == -2 && !v27 )
              v27 = v24;
            v23 = (v53 - 1) & (v26 + v23);
            v24 = &v51[v23];
            v25 = *v24;
            if ( *v24 == v8 )
              goto LABEL_4;
            ++v26;
          }
          if ( !v27 )
            v27 = v24;
          ++v50;
          v28 = v52 + 1;
          if ( 4 * ((int)v52 + 1) < (unsigned int)(3 * v53) )
          {
            if ( (int)v53 - HIDWORD(v52) - v28 <= (unsigned int)v53 >> 3 )
            {
              sub_A08C50((__int64)&v50, v53);
              if ( !(_DWORD)v53 )
              {
LABEL_91:
                LODWORD(v52) = v52 + 1;
                goto LABEL_92;
              }
              v43 = 1;
              v7 = ((_DWORD)v53 - 1) & v47;
              v27 = &v51[v7];
              v28 = v52 + 1;
              v42 = 0;
              v44 = *v27;
              if ( v8 != *v27 )
              {
                while ( v44 != -1 )
                {
                  if ( v44 == -2 && !v42 )
                    v42 = v27;
                  v7 = ((_DWORD)v53 - 1) & (unsigned int)(v43 + v7);
                  v27 = &v51[(unsigned int)v7];
                  v44 = *v27;
                  if ( *v27 == v8 )
                    goto LABEL_31;
                  ++v43;
                }
                goto LABEL_70;
              }
            }
            goto LABEL_31;
          }
LABEL_66:
          sub_A08C50((__int64)&v50, 2 * v53);
          if ( !(_DWORD)v53 )
            goto LABEL_91;
          v7 = ((_DWORD)v53 - 1) & v47;
          v27 = &v51[v7];
          v28 = v52 + 1;
          v40 = *v27;
          if ( v8 != *v27 )
          {
            v41 = 1;
            v42 = 0;
            while ( v40 != -1 )
            {
              if ( v40 == -2 && !v42 )
                v42 = v27;
              v7 = ((_DWORD)v53 - 1) & (unsigned int)(v41 + v7);
              v27 = &v51[(unsigned int)v7];
              v40 = *v27;
              if ( *v27 == v8 )
                goto LABEL_31;
              ++v41;
            }
LABEL_70:
            if ( v42 )
              v27 = v42;
          }
LABEL_31:
          LODWORD(v52) = v28;
          if ( *v27 != -1 )
            --HIDWORD(v52);
          *v27 = v8;
        }
LABEL_4:
        v47 += 37;
        if ( *(_DWORD *)(a2 + 12) <= ++v8 )
          goto LABEL_11;
      }
      ++v46;
      v13 = *(_QWORD *)(v11 + 168);
      if ( !v49 || v13 < v48 )
      {
        v48 = *(_QWORD *)(v11 + 168);
        v49 = 1;
      }
      if ( !v13 )
        goto LABEL_4;
      ++v8;
      sub_FE8630((__int64)v54, v10, v13, 0, v12, v7);
      v47 += 37;
      if ( *(_DWORD *)(a2 + 12) <= v8 )
      {
LABEL_11:
        v3 = a2;
        v14 = v48;
        if ( !v49 )
          v14 = 1;
LABEL_13:
        v15 = v51;
        v16 = &v51[(unsigned int)v53];
        if ( (_DWORD)v52 && v51 != v16 )
        {
          while ( *v15 > 0xFFFFFFFD )
          {
            if ( ++v15 == v16 )
              goto LABEL_14;
          }
          v39 = v54;
          if ( v16 != v15 )
          {
            if ( !v14 )
              goto LABEL_56;
LABEL_60:
            sub_FE8630(
              (__int64)v54,
              (unsigned int *)(*(_QWORD *)(v3 + 96) + 4LL * (unsigned int)*v15),
              v14,
              0,
              (__int64)v39,
              v7);
LABEL_56:
            while ( ++v15 != v16 )
            {
              if ( *v15 <= 0xFFFFFFFD )
              {
                if ( v15 == v16 )
                  break;
                if ( v14 )
                  goto LABEL_60;
              }
            }
          }
        }
LABEL_14:
        sub_FEAD50(a1, (__int64)v54);
        v20 = *(unsigned int **)(v3 + 96);
        v21 = &v20[*(unsigned int *)(v3 + 104)];
        if ( v21 != v20 )
        {
          while ( (unsigned __int8)sub_2E3AA60(a1, v3, v20) )
          {
            if ( v21 == ++v20 )
              goto LABEL_17;
          }
LABEL_92:
          BUG();
        }
LABEL_17:
        if ( !v46 )
          sub_FEAA90(a1, (_DWORD *)v3, v17, v18, v19);
        sub_C7D6A0((__int64)v51, 4LL * (unsigned int)v53, 4);
        if ( (_BYTE *)v54[0] != v55 )
          _libc_free(v54[0]);
LABEL_21:
        sub_FE9590(a1, v3);
        sub_FE86B0(a1, v3);
        return 1;
      }
    }
  }
  v29 = *(_QWORD *)(a1 + 64) + 24LL * **(unsigned int **)(a2 + 96);
  v30 = *(__int64 **)(v29 + 8);
  if ( !v30 )
    goto LABEL_42;
  v31 = *((unsigned int *)v30 + 3);
  v32 = (_DWORD *)v30[12];
  if ( (unsigned int)v31 > 1 )
  {
    if ( !sub_FDC990(v32, &v32[v31], (_DWORD *)v29) )
      goto LABEL_42;
  }
  else if ( *(_DWORD *)v29 != *v32 )
  {
    goto LABEL_42;
  }
  if ( *((_BYTE *)v30 + 8) )
  {
    v33 = *v30;
    if ( !*v30
      || (v34 = *(unsigned int *)(v33 + 12), (unsigned int)v34 <= 1)
      || !sub_FDC990(*(_DWORD **)(v33 + 96), (_DWORD *)(*(_QWORD *)(v33 + 96) + 4 * v34), (_DWORD *)v29)
      || (v35 = (_QWORD *)(v33 + 152), !*(_BYTE *)(v33 + 8)) )
    {
      v35 = v30 + 19;
    }
    goto LABEL_43;
  }
LABEL_42:
  v35 = (_QWORD *)(v29 + 16);
LABEL_43:
  *v35 = -1;
  LODWORD(v54[0]) = **(_DWORD **)(a2 + 96);
  if ( !(unsigned __int8)sub_2E3AA60(a1, a2, (unsigned int *)v54) )
    goto LABEL_92;
  v36 = *(_QWORD *)(a2 + 96);
  v37 = (unsigned int *)(v36 + 4LL * *(unsigned int *)(a2 + 104));
  v38 = (unsigned int *)(v36 + 4LL * *(unsigned int *)(a2 + 12));
  if ( v37 == v38 )
    goto LABEL_21;
  while ( 1 )
  {
    result = sub_2E3AA60(a1, a2, v38);
    if ( !(_BYTE)result )
      return result;
    if ( v37 == ++v38 )
      goto LABEL_21;
  }
}
