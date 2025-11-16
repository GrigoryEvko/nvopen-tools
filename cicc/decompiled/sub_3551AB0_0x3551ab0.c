// Function: sub_3551AB0
// Address: 0x3551ab0
//
__int64 __fastcall sub_3551AB0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 v5; // rdi
  __int64 v6; // rsi
  int *v7; // r15
  unsigned int v9; // esi
  __int64 v10; // rdi
  int v11; // r10d
  int *v12; // r9
  unsigned int v13; // eax
  int *v14; // rdx
  __int64 v15; // rcx
  int *v16; // r13
  int *v17; // r14
  unsigned int v18; // esi
  int v19; // edx
  int *v20; // rcx
  int v21; // edi
  int v22; // ecx
  int v23; // ecx
  __int64 v24; // rsi
  __int64 v25; // rax
  int *v26; // r10
  int v27; // edx
  int *v28; // rbx
  _QWORD *v29; // r13
  __int64 v30; // rax
  __int64 v31; // r14
  _DWORD *v32; // r13
  int v33; // ecx
  __int64 v34; // rdx
  int v35; // r11d
  int v36; // eax
  int v37; // ecx
  __int64 v38; // rdi
  __int64 v39; // [rsp+10h] [rbp-70h]
  int v40; // [rsp+18h] [rbp-68h]
  unsigned __int64 v41; // [rsp+18h] [rbp-68h]
  __int64 v42; // [rsp+20h] [rbp-60h] BYREF
  int *v43; // [rsp+28h] [rbp-58h] BYREF
  __int64 v44; // [rsp+30h] [rbp-50h] BYREF
  int *v45; // [rsp+38h] [rbp-48h]
  __int64 v46; // [rsp+40h] [rbp-40h]
  __int64 v47; // [rsp+48h] [rbp-38h]

  v1 = *a1;
  v44 = 0;
  v45 = 0;
  v2 = *(_QWORD *)(v1 + 56);
  v3 = v1 + 48;
  v46 = 0;
  v47 = 0;
  if ( v2 == v1 + 48 )
  {
    v6 = 0;
    v5 = 0;
    return sub_C7D6A0(v5, v6 * 4, 4);
  }
  v39 = (__int64)(a1 + 10);
  do
  {
    while ( 1 )
    {
      if ( (unsigned __int16)(*(_WORD *)(v2 + 68) - 14) <= 4u )
        goto LABEL_5;
      v9 = *((_DWORD *)a1 + 26);
      v42 = v2;
      if ( !v9 )
      {
        ++a1[10];
        v43 = 0;
LABEL_78:
        v9 *= 2;
LABEL_79:
        sub_354EAF0(v39, v9);
        sub_35473A0(v39, &v42, &v43);
        v38 = v42;
        v14 = v43;
        v37 = *((_DWORD *)a1 + 24) + 1;
        goto LABEL_74;
      }
      v10 = a1[11];
      v11 = 1;
      v12 = 0;
      v13 = (v9 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v14 = (int *)(v10 + 632LL * v13);
      v15 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 == v2 )
      {
LABEL_14:
        v16 = (int *)*((_QWORD *)v14 + 1);
        v17 = &v16[6 * v14[4]];
        if ( v17 == v16 )
          goto LABEL_5;
        while ( 1 )
        {
          v22 = *v16;
          LODWORD(v42) = *v16;
          if ( *(_WORD *)(v2 + 68) == 68 || !*(_WORD *)(v2 + 68) )
          {
            v23 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
            if ( v23 == 1 )
            {
LABEL_35:
              if ( (_DWORD)v42 )
                goto LABEL_21;
              goto LABEL_19;
            }
            v24 = *(_QWORD *)(v2 + 32);
            v25 = 1;
            while ( *a1 != *(_QWORD *)(v24 + 40LL * (unsigned int)(v25 + 1) + 24) )
            {
              v25 = (unsigned int)(v25 + 2);
              if ( v23 == (_DWORD)v25 )
                goto LABEL_35;
            }
            v22 = *(_DWORD *)(v24 + 40 * v25 + 8);
            if ( (_DWORD)v42 != v22 )
              goto LABEL_21;
            if ( (unsigned int)(v22 - 1) <= 0x3FFFFFFE )
            {
LABEL_18:
              if ( (*(_QWORD *)(*(_QWORD *)(a1[1] + 384) + 8LL * ((unsigned int)v22 >> 6)) & (1LL << v22)) != 0 )
                goto LABEL_21;
              goto LABEL_19;
            }
          }
          else if ( (unsigned int)(v22 - 1) <= 0x3FFFFFFE )
          {
            goto LABEL_18;
          }
          if ( v22 < 0 )
          {
            if ( *(_QWORD *)(sub_2EBEE10(a1[1], v42) + 24) == *a1 )
              goto LABEL_21;
            v18 = v47;
            if ( !(_DWORD)v47 )
            {
LABEL_32:
              ++v44;
              v43 = 0;
              goto LABEL_33;
            }
            goto LABEL_20;
          }
LABEL_19:
          v18 = v47;
          if ( !(_DWORD)v47 )
            goto LABEL_32;
LABEL_20:
          v19 = (v18 - 1) & (37 * v42);
          v20 = &v45[v19];
          v21 = *v20;
          if ( *v20 != (_DWORD)v42 )
          {
            v35 = 1;
            v26 = 0;
            while ( v21 != -1 )
            {
              if ( v21 == -2 && !v26 )
                v26 = v20;
              v19 = (v18 - 1) & (v35 + v19);
              v20 = &v45[v19];
              v21 = *v20;
              if ( (_DWORD)v42 == *v20 )
                goto LABEL_21;
              ++v35;
            }
            if ( !v26 )
              v26 = v20;
            ++v44;
            v27 = v46 + 1;
            v43 = v26;
            if ( 4 * ((int)v46 + 1) >= 3 * v18 )
            {
LABEL_33:
              v18 *= 2;
            }
            else if ( v18 - HIDWORD(v46) - v27 > v18 >> 3 )
            {
              goto LABEL_61;
            }
            sub_2E29BA0((__int64)&v44, v18);
            sub_3549AD0((__int64)&v44, (int *)&v42, &v43);
            v26 = v43;
            v27 = v46 + 1;
LABEL_61:
            LODWORD(v46) = v27;
            if ( *v26 != -1 )
              --HIDWORD(v46);
            *v26 = v42;
          }
LABEL_21:
          v16 += 6;
          if ( v17 == v16 )
            goto LABEL_5;
        }
      }
      while ( v15 != -4096 )
      {
        if ( v15 == -8192 && !v12 )
          v12 = v14;
        v13 = (v9 - 1) & (v11 + v13);
        v14 = (int *)(v10 + 632LL * v13);
        v15 = *(_QWORD *)v14;
        if ( *(_QWORD *)v14 == v2 )
          goto LABEL_14;
        ++v11;
      }
      v36 = *((_DWORD *)a1 + 24);
      if ( v12 )
        v14 = v12;
      ++a1[10];
      v37 = v36 + 1;
      v43 = v14;
      if ( 4 * (v36 + 1) >= 3 * v9 )
        goto LABEL_78;
      v38 = v2;
      if ( v9 - *((_DWORD *)a1 + 25) - v37 <= v9 >> 3 )
        goto LABEL_79;
LABEL_74:
      *((_DWORD *)a1 + 24) = v37;
      if ( *(_QWORD *)v14 != -4096 )
        --*((_DWORD *)a1 + 25);
      *(_QWORD *)v14 = v38;
      memset(v14 + 2, 0, 0x270u);
      *((_QWORD *)v14 + 1) = v14 + 6;
      *((_QWORD *)v14 + 27) = v14 + 58;
      *((_QWORD *)v14 + 2) = 0x800000000LL;
      *((_QWORD *)v14 + 28) = 0x800000000LL;
      *((_QWORD *)v14 + 53) = v14 + 110;
      *((_QWORD *)v14 + 54) = 0x800000000LL;
LABEL_5:
      if ( (*(_BYTE *)v2 & 4) == 0 )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        goto LABEL_7;
    }
    while ( (*(_BYTE *)(v2 + 44) & 8) != 0 )
      v2 = *(_QWORD *)(v2 + 8);
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v3 != v2 );
LABEL_7:
  v5 = (__int64)v45;
  v6 = (unsigned int)v47;
  v7 = &v45[v6];
  if ( (_DWORD)v46 && v7 != v45 )
  {
    v28 = v45;
    while ( (unsigned int)*v28 > 0xFFFFFFFD )
    {
      if ( v7 == ++v28 )
        return sub_C7D6A0(v5, v6 * 4, 4);
    }
    if ( v7 != v28 )
    {
      do
      {
        v29 = (_QWORD *)a1[1];
        v40 = *v28;
        v30 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v29 + 16LL) + 200LL))(*(_QWORD *)(*v29 + 16LL));
        v31 = v30;
        if ( v40 < 0 )
        {
          v41 = *(_QWORD *)(v29[7] + 16LL * (v40 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
          v32 = (_DWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v30 + 416LL))(v30);
          v33 = *(_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v31 + 376LL))(v31, v41);
        }
        else
        {
          v32 = (_DWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v30 + 424LL))(v30);
          v33 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v31 + 384LL))(v31, (unsigned int)v40);
        }
        if ( *v32 == -1 )
          v32 = 0;
        do
        {
          if ( !v32 )
            break;
          v34 = (unsigned int)*v32++;
          *(_DWORD *)(a1[4] + 4 * v34) += v33;
        }
        while ( *v32 != -1 );
        if ( ++v28 == v7 )
          break;
        while ( (unsigned int)*v28 > 0xFFFFFFFD )
        {
          if ( v7 == ++v28 )
            goto LABEL_51;
        }
      }
      while ( v7 != v28 );
LABEL_51:
      v5 = (__int64)v45;
      v6 = (unsigned int)v47;
    }
  }
  return sub_C7D6A0(v5, v6 * 4, 4);
}
