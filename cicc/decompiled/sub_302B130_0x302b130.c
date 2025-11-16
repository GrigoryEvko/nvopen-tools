// Function: sub_302B130
// Address: 0x302b130
//
__int64 __fastcall sub_302B130(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v5; // rdi
  unsigned int v6; // r8d
  __int16 ***v7; // r12
  int v8; // r11d
  __int64 v9; // r9
  unsigned int v10; // ecx
  __int64 *v11; // r13
  __int16 ****v12; // rax
  __int16 ***v13; // rdx
  unsigned int v14; // r8d
  __int64 v15; // r9
  __int64 v16; // rbx
  int v17; // r11d
  _DWORD *v18; // r10
  unsigned int v19; // eax
  _DWORD *v20; // rdx
  int v21; // ecx
  int v22; // eax
  int v23; // edx
  int v24; // ecx
  int v25; // ecx
  int v26; // edx
  int *v27; // rax
  int v28; // eax
  int v29; // eax
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // edx
  __int16 ***v33; // rdi
  int v34; // r10d
  __int16 ****v35; // r9
  int v36; // eax
  int v37; // edx
  __int64 v38; // rdi
  __int16 ****v39; // r8
  unsigned int v40; // r14d
  int v41; // r9d
  __int16 ***v42; // rsi
  int v43[3]; // [rsp+Ch] [rbp-44h] BYREF
  _QWORD v44[7]; // [rsp+18h] [rbp-38h] BYREF

  result = a2 & 0xFFFFFFF;
  v43[0] = a2;
  if ( a2 >= 0 )
    return result;
  v3 = *(_QWORD *)(a1 + 1104);
  v5 = a1 + 1112;
  v6 = *(_DWORD *)(a1 + 1136);
  v7 = (__int16 ***)(*(_QWORD *)(*(_QWORD *)(v3 + 56) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 1112);
    goto LABEL_52;
  }
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 1120);
  v10 = (v6 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v11 = (__int64 *)(v9 + 40LL * v10);
  v12 = 0;
  v13 = (__int16 ***)*v11;
  if ( v7 != (__int16 ***)*v11 )
  {
    while ( v13 != (__int16 ***)-4096LL )
    {
      if ( !v12 && v13 == (__int16 ***)-8192LL )
        v12 = (__int16 ****)v11;
      v10 = (v6 - 1) & (v8 + v10);
      v11 = (__int64 *)(v9 + 40LL * v10);
      v13 = (__int16 ***)*v11;
      if ( v7 == (__int16 ***)*v11 )
        goto LABEL_4;
      ++v8;
    }
    v24 = *(_DWORD *)(a1 + 1128);
    if ( !v12 )
      v12 = (__int16 ****)v11;
    ++*(_QWORD *)(a1 + 1112);
    v25 = v24 + 1;
    if ( 4 * v25 < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 1132) - v25 > v6 >> 3 )
      {
LABEL_29:
        *(_DWORD *)(a1 + 1128) = v25;
        if ( *v12 != (__int16 ***)-4096LL )
          --*(_DWORD *)(a1 + 1132);
        *v12 = v7;
        v16 = (__int64)(v12 + 1);
        v12[1] = 0;
        v12[2] = 0;
        v12[3] = 0;
        *((_DWORD *)v12 + 8) = 0;
        goto LABEL_32;
      }
      sub_302AEB0(v5, v6);
      v36 = *(_DWORD *)(a1 + 1136);
      if ( v36 )
      {
        v37 = v36 - 1;
        v38 = *(_QWORD *)(a1 + 1120);
        v39 = 0;
        v40 = (v36 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v41 = 1;
        v25 = *(_DWORD *)(a1 + 1128) + 1;
        v12 = (__int16 ****)(v38 + 40LL * v40);
        v42 = *v12;
        if ( v7 != *v12 )
        {
          while ( v42 != (__int16 ***)-4096LL )
          {
            if ( !v39 && v42 == (__int16 ***)-8192LL )
              v39 = v12;
            v40 = v37 & (v41 + v40);
            v12 = (__int16 ****)(v38 + 40LL * v40);
            v42 = *v12;
            if ( v7 == *v12 )
              goto LABEL_29;
            ++v41;
          }
          if ( v39 )
            v12 = v39;
        }
        goto LABEL_29;
      }
LABEL_76:
      ++*(_DWORD *)(a1 + 1128);
      BUG();
    }
LABEL_52:
    sub_302AEB0(v5, 2 * v6);
    v29 = *(_DWORD *)(a1 + 1136);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 1120);
      v32 = (v29 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v25 = *(_DWORD *)(a1 + 1128) + 1;
      v12 = (__int16 ****)(v31 + 40LL * v32);
      v33 = *v12;
      if ( v7 != *v12 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != (__int16 ***)-4096LL )
        {
          if ( v33 == (__int16 ***)-8192LL && !v35 )
            v35 = v12;
          v32 = v30 & (v34 + v32);
          v12 = (__int16 ****)(v31 + 40LL * v32);
          v33 = *v12;
          if ( v7 == *v12 )
            goto LABEL_29;
          ++v34;
        }
        if ( v35 )
          v12 = v35;
      }
      goto LABEL_29;
    }
    goto LABEL_76;
  }
LABEL_4:
  v14 = *((_DWORD *)v11 + 8);
  v15 = v11[2];
  v16 = (__int64)(v11 + 1);
  if ( !v14 )
  {
LABEL_32:
    v44[0] = 0;
    v14 = 0;
    ++*(_QWORD *)v16;
    goto LABEL_33;
  }
  v17 = 1;
  v18 = 0;
  v19 = (v14 - 1) & (37 * a2);
  v20 = (_DWORD *)(v15 + 8LL * v19);
  v21 = *v20;
  if ( a2 != *v20 )
  {
    while ( v21 != -1 )
    {
      if ( !v18 && v21 == -2 )
        v18 = v20;
      v19 = (v14 - 1) & (v17 + v19);
      v20 = (_DWORD *)(v15 + 8LL * v19);
      v21 = *v20;
      if ( a2 == *v20 )
        goto LABEL_6;
      ++v17;
    }
    v28 = *((_DWORD *)v11 + 6);
    if ( !v18 )
      v18 = v20;
    ++v11[1];
    v26 = v28 + 1;
    v44[0] = v18;
    if ( 4 * (v28 + 1) < 3 * v14 )
    {
      if ( v14 - *((_DWORD *)v11 + 7) - v26 <= v14 >> 3 )
      {
        sub_A09770((__int64)(v11 + 1), v14);
        sub_A1A0F0((__int64)(v11 + 1), v43, v44);
        a2 = v43[0];
        v26 = *((_DWORD *)v11 + 6) + 1;
      }
      goto LABEL_34;
    }
LABEL_33:
    sub_A09770(v16, 2 * v14);
    sub_A1A0F0(v16, v43, v44);
    a2 = v43[0];
    v26 = *(_DWORD *)(v16 + 16) + 1;
LABEL_34:
    v27 = (int *)v44[0];
    *(_DWORD *)(v16 + 16) = v26;
    if ( *v27 != -1 )
      --*(_DWORD *)(v16 + 20);
    *v27 = a2;
    v27[1] = 0;
    v22 = 0;
    goto LABEL_7;
  }
LABEL_6:
  v22 = v20[1];
LABEL_7:
  if ( v7 == &off_4A2FD40 )
  {
    v23 = 0x10000000;
  }
  else if ( v7 == &off_4A2FCE0 )
  {
    v23 = 0x20000000;
  }
  else if ( v7 == &off_4A2FC20 )
  {
    v23 = 805306368;
  }
  else if ( v7 == &off_4A2FA40 )
  {
    v23 = 0x40000000;
  }
  else if ( v7 == &off_4A2FB60 )
  {
    v23 = 1342177280;
  }
  else if ( v7 == &off_4A2F980 )
  {
    v23 = 1610612736;
  }
  else
  {
    if ( v7 != &off_4A2F8C0 )
      sub_C64ED0("Bad register class", 1u);
    v23 = 1879048192;
  }
  return v23 | v22 & 0xFFFFFFFu;
}
