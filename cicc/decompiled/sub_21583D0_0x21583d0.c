// Function: sub_21583D0
// Address: 0x21583d0
//
__int64 __fastcall sub_21583D0(__int64 a1, int a2)
{
  unsigned int v2; // r8d
  int v4; // eax
  unsigned int v6; // r8d
  __int64 v7; // r13
  unsigned __int64 v8; // rdx
  __int16 ***v9; // rsi
  __int64 v10; // r9
  unsigned int v11; // ecx
  __int16 ****v12; // rbx
  __int16 ***v13; // rdi
  unsigned int v14; // r8d
  __int16 ***v15; // r10
  __int64 v16; // r12
  unsigned int v17; // edx
  int *v18; // rcx
  int v19; // edi
  int v20; // r8d
  unsigned int v21; // eax
  int v22; // r13d
  int *v23; // r9
  int v24; // ecx
  int v25; // ecx
  int v26; // r14d
  __int16 ****v27; // r11
  int v28; // eax
  int v29; // edx
  __int64 v30; // rax
  int v31; // esi
  int *v32; // r14
  int v33; // [rsp+Ch] [rbp-44h] BYREF
  __int16 ***v34; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a2 & 0xFFFFFFF;
  v33 = a2;
  if ( a2 >= 0 )
    return v2;
  v4 = a2;
  v6 = *(_DWORD *)(a1 + 832);
  v7 = a1 + 808;
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 800) + 24LL) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v34 = (__int16 ***)v8;
  v9 = (__int16 ***)v8;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 808);
    goto LABEL_48;
  }
  v10 = *(_QWORD *)(a1 + 816);
  v11 = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v12 = (__int16 ****)(v10 + 40LL * v11);
  v13 = *v12;
  if ( (__int16 ***)v8 != *v12 )
  {
    v26 = 1;
    v27 = 0;
    while ( v13 != (__int16 ***)-8LL )
    {
      if ( v13 == (__int16 ***)-16LL && !v27 )
        v27 = v12;
      v11 = (v6 - 1) & (v26 + v11);
      v12 = (__int16 ****)(v10 + 40LL * v11);
      v13 = *v12;
      if ( (__int16 ***)v8 == *v12 )
        goto LABEL_5;
      ++v26;
    }
    v28 = *(_DWORD *)(a1 + 824);
    if ( v27 )
      v12 = v27;
    ++*(_QWORD *)(a1 + 808);
    v29 = v28 + 1;
    if ( 4 * (v28 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 828) - v29 > v6 >> 3 )
      {
LABEL_41:
        *(_DWORD *)(a1 + 824) = v29;
        if ( *v12 != (__int16 ***)-8LL )
          --*(_DWORD *)(a1 + 828);
        *v12 = v9;
        v16 = (__int64)(v12 + 1);
        v30 = 1;
        v12[1] = 0;
        v12[2] = 0;
        v12[3] = 0;
        *((_DWORD *)v12 + 8) = 0;
        goto LABEL_44;
      }
      sub_2158190(v7, v6);
LABEL_49:
      sub_2155090(v7, (__int64 *)&v34, v35);
      v12 = (__int16 ****)v35[0];
      v9 = v34;
      v29 = *(_DWORD *)(a1 + 824) + 1;
      goto LABEL_41;
    }
LABEL_48:
    sub_2158190(v7, 2 * v6);
    goto LABEL_49;
  }
LABEL_5:
  v14 = *((_DWORD *)v12 + 8);
  v15 = v12[2];
  v16 = (__int64)(v12 + 1);
  if ( !v14 )
  {
    v30 = (__int64)v12[1] + 1;
LABEL_44:
    v12[1] = (__int16 ***)v30;
    v31 = 0;
    goto LABEL_45;
  }
  v17 = (v14 - 1) & (37 * v4);
  v18 = (int *)&v15[v17];
  v19 = *v18;
  if ( v4 != *v18 )
  {
    v22 = 1;
    v23 = 0;
    while ( v19 != -1 )
    {
      if ( v23 || v19 != -2 )
        v18 = v23;
      v17 = (v14 - 1) & (v22 + v17);
      v32 = (int *)&v15[v17];
      v19 = *v32;
      if ( v4 == *v32 )
      {
        v20 = v32[1];
        goto LABEL_8;
      }
      ++v22;
      v23 = v18;
      v18 = (int *)&v15[v17];
    }
    if ( !v23 )
      v23 = v18;
    v24 = *((_DWORD *)v12 + 6);
    v12[1] = (__int16 ***)((char *)v12[1] + 1);
    v25 = v24 + 1;
    if ( 4 * v25 < 3 * v14 )
    {
      if ( v14 - *((_DWORD *)v12 + 7) - v25 <= v14 >> 3 )
      {
        sub_1392B70((__int64)(v12 + 1), v14);
        sub_1932870((__int64)(v12 + 1), &v33, v35);
        v23 = (int *)v35[0];
        v4 = v33;
        v25 = *((_DWORD *)v12 + 6) + 1;
      }
      goto LABEL_32;
    }
    v31 = 2 * v14;
LABEL_45:
    sub_1392B70(v16, v31);
    sub_1932870(v16, &v33, v35);
    v23 = (int *)v35[0];
    v4 = v33;
    v25 = *((_DWORD *)v12 + 6) + 1;
LABEL_32:
    *((_DWORD *)v12 + 6) = v25;
    if ( *v23 != -1 )
      --*((_DWORD *)v12 + 7);
    *v23 = v4;
    v20 = 0;
    v23[1] = 0;
    v9 = v34;
    goto LABEL_8;
  }
  v20 = v18[1];
LABEL_8:
  if ( v9 == &off_4A027A0 )
  {
    v21 = 0x10000000;
  }
  else if ( v9 == &off_4A02720 )
  {
    v21 = 0x20000000;
  }
  else if ( v9 == &off_4A025A0 )
  {
    v21 = 805306368;
  }
  else if ( v9 == &off_4A024A0 )
  {
    v21 = 0x40000000;
  }
  else if ( v9 == &off_4A02620 )
  {
    v21 = 1342177280;
  }
  else if ( v9 == &off_4A02520 )
  {
    v21 = 1610612736;
  }
  else if ( v9 == &off_4A02760 )
  {
    v21 = 1879048192;
  }
  else if ( v9 == &off_4A026A0 )
  {
    v21 = 0x80000000;
  }
  else
  {
    if ( v9 != &off_4A02460 )
      sub_16BD130("Bad register class", 1u);
    v21 = -1879048192;
  }
  return v21 | v20 & 0xFFFFFFF;
}
