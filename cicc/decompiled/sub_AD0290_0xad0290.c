// Function: sub_AD0290
// Address: 0xad0290
//
__int64 __fastcall sub_AD0290(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // r13
  __int64 *v8; // r14
  int v9; // ebx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rsi
  unsigned int v13; // r8d
  __int64 *v14; // rcx
  __int64 v15; // rdx
  int v16; // r11d
  __int64 i; // rdi
  __int64 result; // rax
  __int64 v19; // r8
  unsigned int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // edi
  __int64 *v23; // rcx
  __int64 v24; // rdx
  int v25; // edx
  int v26; // edx
  __int64 v27; // rcx
  int v28; // esi
  int v29; // esi
  __int64 v30; // r9
  unsigned int v31; // ebx
  __int64 v32; // rdi
  int v33; // r11d
  __int64 *v34; // r8
  int v35; // esi
  int v36; // esi
  __int64 v37; // r9
  unsigned int v38; // ebx
  __int64 v39; // rdi
  int v40; // r11d
  __int64 v41; // r10
  __int64 v42; // r10
  __int64 v43; // [rsp+0h] [rbp-A0h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  __int64 v45; // [rsp+10h] [rbp-90h]
  int v46; // [rsp+10h] [rbp-90h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+18h] [rbp-88h]
  __int64 *v49; // [rsp+18h] [rbp-88h]
  __int64 v50; // [rsp+18h] [rbp-88h]
  __int64 v51; // [rsp+18h] [rbp-88h]
  __int64 *v52; // [rsp+18h] [rbp-88h]
  int v53; // [rsp+2Ch] [rbp-74h] BYREF
  __int64 v54; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v55; // [rsp+38h] [rbp-68h]
  __int64 v56; // [rsp+40h] [rbp-60h]
  __int64 v57; // [rsp+50h] [rbp-50h] BYREF
  __int64 v58; // [rsp+58h] [rbp-48h]
  __int64 v59; // [rsp+60h] [rbp-40h]
  __int64 v60; // [rsp+68h] [rbp-38h]
  char v61[48]; // [rsp+70h] [rbp-30h] BYREF

  v59 = a3;
  v60 = a4;
  v57 = a1;
  v58 = a2;
  v4 = *(_QWORD *)sub_BD5C60(a1, a2, a3);
  v5 = *(_QWORD *)(a1 + 8);
  v55 = &v57;
  v56 = 4;
  v54 = v5;
  v53 = sub_AC5F60(&v57, (__int64)v61);
  v6 = sub_AC7AE0(&v54, &v53);
  v7 = v54;
  v8 = v55;
  v9 = v6;
  v10 = *(unsigned int *)(v4 + 2112);
  v11 = v56;
  v12 = *(_QWORD *)(v4 + 2096);
  if ( (_DWORD)v10 )
  {
    v13 = v9 & (v10 - 1);
    v14 = (__int64 *)(v12 + 8LL * v13);
    v15 = *v14;
    if ( *v14 != -4096 )
    {
      v16 = 1;
      while ( v15 == -8192 || *(_QWORD *)(v15 + 8) != v54 || v56 != 4 )
      {
LABEL_6:
        v13 = (v10 - 1) & (v16 + v13);
        v14 = (__int64 *)(v12 + 8LL * v13);
        v15 = *v14;
        if ( *v14 == -4096 )
          goto LABEL_14;
        ++v16;
      }
      for ( i = 0; i != 4; ++i )
      {
        if ( v55[i] != *(_QWORD *)(v15 + 32 * i - 128) )
          goto LABEL_6;
      }
      if ( v14 != (__int64 *)(v12 + 8 * v10) )
        return *v14;
    }
  }
LABEL_14:
  v43 = v57;
  v44 = v58;
  v45 = v59;
  v47 = v60;
  result = sub_BD2C40(24, unk_3F28994);
  if ( result )
  {
    v19 = v47;
    v48 = result;
    sub_AC43C0(result, v43, v44, v45, v19);
    result = v48;
  }
  v20 = *(_DWORD *)(v4 + 2112);
  if ( !v20 )
  {
    ++*(_QWORD *)(v4 + 2088);
    goto LABEL_43;
  }
  v21 = *(_QWORD *)(v4 + 2096);
  v22 = v9 & (v20 - 1);
  v23 = (__int64 *)(v21 + 8LL * v22);
  v24 = *v23;
  if ( *v23 != -4096 )
  {
    v46 = 1;
    v49 = 0;
    while ( 1 )
    {
      if ( v24 == -8192 )
      {
        if ( v49 )
          v23 = v49;
        v49 = v23;
      }
      else if ( v7 == *(_QWORD *)(v24 + 8) && v11 == 4 )
      {
        v27 = 0;
        while ( v8[v27] == *(_QWORD *)(v24 + 32 * v27 - 128) )
        {
          if ( ++v27 == 4 )
            return result;
        }
      }
      v22 = (v20 - 1) & (v46 + v22);
      v23 = (__int64 *)(v21 + 8LL * v22);
      v24 = *v23;
      if ( *v23 == -4096 )
        break;
      ++v46;
    }
    if ( v49 )
      v23 = v49;
  }
  v25 = *(_DWORD *)(v4 + 2104);
  ++*(_QWORD *)(v4 + 2088);
  v26 = v25 + 1;
  if ( 4 * v26 >= 3 * v20 )
  {
LABEL_43:
    v51 = result;
    sub_AD00B0(v4 + 2088, 2 * v20);
    v35 = *(_DWORD *)(v4 + 2112);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(v4 + 2096);
      v38 = v36 & v9;
      v23 = (__int64 *)(v37 + 8LL * v38);
      v26 = *(_DWORD *)(v4 + 2104) + 1;
      result = v51;
      v39 = *v23;
      if ( *v23 == -4096 )
        goto LABEL_27;
      v40 = 1;
      v34 = 0;
      while ( 1 )
      {
        if ( v39 == -8192 )
        {
          if ( !v34 )
            v34 = v23;
        }
        else if ( v7 == *(_QWORD *)(v39 + 8) && v11 == 4 )
        {
          v52 = v23;
          v41 = 0;
          while ( v8[v41] == *(_QWORD *)(v39 + 32 * v41 - 128) )
          {
            if ( ++v41 == 4 )
            {
LABEL_54:
              v23 = v52;
              goto LABEL_27;
            }
          }
        }
        v38 = v36 & (v40 + v38);
        v23 = (__int64 *)(v37 + 8LL * v38);
        v39 = *v23;
        if ( *v23 == -4096 )
          goto LABEL_66;
        ++v40;
      }
    }
LABEL_72:
    ++*(_DWORD *)(v4 + 2104);
    BUG();
  }
  if ( v20 - *(_DWORD *)(v4 + 2108) - v26 <= v20 >> 3 )
  {
    v50 = result;
    sub_AD00B0(v4 + 2088, v20);
    v28 = *(_DWORD *)(v4 + 2112);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(v4 + 2096);
      v31 = v29 & v9;
      v23 = (__int64 *)(v30 + 8LL * v31);
      v26 = *(_DWORD *)(v4 + 2104) + 1;
      result = v50;
      v32 = *v23;
      if ( *v23 == -4096 )
        goto LABEL_27;
      v33 = 1;
      v34 = 0;
      while ( 1 )
      {
        if ( v32 == -8192 )
        {
          if ( !v34 )
            v34 = v23;
        }
        else if ( v7 == *(_QWORD *)(v32 + 8) && v11 == 4 )
        {
          v52 = v23;
          v42 = 0;
          while ( v8[v42] == *(_QWORD *)(v32 + 32 * v42 - 128) )
          {
            if ( ++v42 == 4 )
              goto LABEL_54;
          }
        }
        v31 = v29 & (v33 + v31);
        v23 = (__int64 *)(v30 + 8LL * v31);
        v32 = *v23;
        if ( *v23 == -4096 )
          break;
        ++v33;
      }
LABEL_66:
      if ( v34 )
        v23 = v34;
      goto LABEL_27;
    }
    goto LABEL_72;
  }
LABEL_27:
  *(_DWORD *)(v4 + 2104) = v26;
  if ( *v23 != -4096 )
    --*(_DWORD *)(v4 + 2108);
  *v23 = result;
  return result;
}
