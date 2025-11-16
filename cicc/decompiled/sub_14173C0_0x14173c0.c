// Function: sub_14173C0
// Address: 0x14173c0
//
__int64 __fastcall sub_14173C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 result; // rax
  _QWORD *v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r13
  char v14; // al
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r14
  unsigned int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rdi
  unsigned int v24; // esi
  __int64 v25; // rcx
  __int64 v26; // rdi
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // r9
  int v30; // r11d
  __int64 *v31; // r10
  int v32; // edx
  int v33; // edx
  __int64 v34; // rdi
  int v35; // r11d
  __int64 *v36; // r10
  int v37; // edx
  int v38; // edx
  __int64 v40; // [rsp+18h] [rbp-A8h]
  __int64 v41; // [rsp+18h] [rbp-A8h]
  __int64 v42; // [rsp+28h] [rbp-98h] BYREF
  __int64 v43; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v44; // [rsp+38h] [rbp-88h] BYREF
  _QWORD *v45; // [rsp+40h] [rbp-80h] BYREF
  __int64 v46; // [rsp+48h] [rbp-78h]
  _QWORD v47[14]; // [rsp+50h] [rbp-70h] BYREF

  if ( !*(_QWORD *)(a2 + 48) && *(__int16 *)(a2 + 18) >= 0 )
    return 0x6000000000000003LL;
  if ( !sub_1625790(a2, 16) )
    return 0x6000000000000003LL;
  v5 = sub_1649C60(*(_QWORD *)(a2 - 24));
  if ( *(_BYTE *)(v5 + 16) <= 3u )
    return 0x6000000000000003LL;
  v7 = v47;
  v47[0] = v5;
  v45 = v47;
  v42 = 0;
  v46 = 0x800000001LL;
  v8 = 1;
  while ( 1 )
  {
    v9 = v8--;
    v10 = v7[v9 - 1];
    LODWORD(v46) = v8;
    v11 = *(_QWORD *)(v10 + 8);
    if ( v11 )
    {
      while ( 1 )
      {
        v12 = sub_1648700(v11);
        v13 = v12;
        if ( *(_BYTE *)(v12 + 16) <= 0x17u || v12 == a2 || !(unsigned __int8)sub_15CCEE0(*(_QWORD *)(a1 + 280), v12, a2) )
          goto LABEL_11;
        v14 = *(_BYTE *)(v13 + 16);
        if ( v14 == 71 )
          goto LABEL_29;
        if ( v14 == 56 )
        {
          if ( (unsigned __int8)sub_15FA1F0(v13) )
          {
LABEL_29:
            v16 = (unsigned int)v46;
            if ( (unsigned int)v46 >= HIDWORD(v46) )
            {
              sub_16CD150(&v45, v47, 0, 8);
              v16 = (unsigned int)v46;
            }
            v45[v16] = v13;
            LODWORD(v46) = v46 + 1;
            goto LABEL_11;
          }
          v14 = *(_BYTE *)(v13 + 16);
        }
        if ( (unsigned __int8)(v14 - 54) <= 1u
          && (*(_QWORD *)(v13 + 48) || *(__int16 *)(v13 + 18) < 0)
          && sub_1625790(v13, 16) )
        {
          v15 = v42;
          if ( v42 )
          {
            if ( (unsigned __int8)sub_15CCEE0(*(_QWORD *)(a1 + 280), v42, v13) )
              v15 = v13;
          }
          else
          {
            v15 = v13;
          }
          v11 = *(_QWORD *)(v11 + 8);
          v42 = v15;
          if ( !v11 )
          {
LABEL_26:
            v8 = v46;
            break;
          }
        }
        else
        {
LABEL_11:
          v11 = *(_QWORD *)(v11 + 8);
          if ( !v11 )
            goto LABEL_26;
        }
      }
    }
    if ( !v8 )
      break;
    v7 = v45;
  }
  result = 0x6000000000000003LL;
  if ( !v42 )
    goto LABEL_42;
  v17 = *(_QWORD *)(v42 + 40);
  v18 = v42 | 2;
  result = v42 | 2;
  if ( a3 == v17 )
    goto LABEL_42;
  v19 = *(_DWORD *)(a1 + 56);
  v43 = a2;
  if ( !v19 )
  {
    ++*(_QWORD *)(a1 + 32);
LABEL_68:
    v41 = v17;
    v19 *= 2;
    goto LABEL_66;
  }
  v20 = *(_QWORD *)(a1 + 40);
  v21 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v22 = (__int64 *)(v20 + 32LL * v21);
  v23 = *v22;
  if ( a2 == *v22 )
    goto LABEL_39;
  v30 = 1;
  v31 = 0;
  while ( v23 != -8 )
  {
    if ( !v31 && v23 == -16 )
      v31 = v22;
    v21 = (v19 - 1) & (v30 + v21);
    v22 = (__int64 *)(v20 + 32LL * v21);
    v23 = *v22;
    if ( a2 == *v22 )
      goto LABEL_39;
    ++v30;
  }
  v32 = *(_DWORD *)(a1 + 48);
  if ( v31 )
    v22 = v31;
  ++*(_QWORD *)(a1 + 32);
  v33 = v32 + 1;
  if ( 4 * v33 >= 3 * v19 )
    goto LABEL_68;
  v34 = a2;
  if ( v19 - *(_DWORD *)(a1 + 52) - v33 <= v19 >> 3 )
  {
    v41 = v17;
LABEL_66:
    sub_14155E0(a1 + 32, v19);
    sub_1414A50(a1 + 32, &v43, &v44);
    v22 = v44;
    v34 = v43;
    v33 = *(_DWORD *)(a1 + 48) + 1;
    v17 = v41;
  }
  *(_DWORD *)(a1 + 48) = v33;
  if ( *v22 != -8 )
    --*(_DWORD *)(a1 + 52);
  *v22 = v34;
  v22[1] = v17;
  v22[2] = v18;
  v22[3] = 0;
LABEL_39:
  v24 = *(_DWORD *)(a1 + 88);
  if ( !v24 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_63;
  }
  v25 = v42;
  v26 = *(_QWORD *)(a1 + 72);
  v27 = (v24 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
  v28 = (__int64 *)(v26 + 80LL * v27);
  v29 = *v28;
  if ( *v28 != v42 )
  {
    v35 = 1;
    v36 = 0;
    while ( v29 != -8 )
    {
      if ( !v36 && v29 == -16 )
        v36 = v28;
      v27 = (v24 - 1) & (v35 + v27);
      v28 = (__int64 *)(v26 + 80LL * v27);
      v29 = *v28;
      if ( v42 == *v28 )
        goto LABEL_41;
      ++v35;
    }
    v37 = *(_DWORD *)(a1 + 80);
    if ( v36 )
      v28 = v36;
    ++*(_QWORD *)(a1 + 64);
    v38 = v37 + 1;
    if ( 4 * v38 < 3 * v24 )
    {
      if ( v24 - *(_DWORD *)(a1 + 84) - v38 > v24 >> 3 )
      {
LABEL_59:
        *(_DWORD *)(a1 + 80) = v38;
        if ( *v28 != -8 )
          --*(_DWORD *)(a1 + 84);
        *v28 = v25;
        v28[1] = 0;
        v28[2] = (__int64)(v28 + 6);
        v28[3] = (__int64)(v28 + 6);
        v28[4] = 4;
        *((_DWORD *)v28 + 10) = 0;
        goto LABEL_41;
      }
LABEL_64:
      sub_14171D0(a1 + 64, v24);
      sub_14153C0(a1 + 64, &v42, &v44);
      v28 = v44;
      v25 = v42;
      v38 = *(_DWORD *)(a1 + 80) + 1;
      goto LABEL_59;
    }
LABEL_63:
    v24 *= 2;
    goto LABEL_64;
  }
LABEL_41:
  sub_1412190((__int64)(v28 + 1), a2);
  result = 0x2000000000000003LL;
LABEL_42:
  if ( v45 != v47 )
  {
    v40 = result;
    _libc_free((unsigned __int64)v45);
    return v40;
  }
  return result;
}
