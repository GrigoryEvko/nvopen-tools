// Function: sub_10643D0
// Address: 0x10643d0
//
__int64 __fastcall sub_10643D0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r13
  __int64 *v4; // r12
  __int64 *i; // rbx
  __int64 v6; // rsi
  __int64 *v7; // rbx
  __int64 *v8; // r12
  unsigned int v10; // esi
  __int64 v11; // r13
  __int64 v12; // r9
  int v13; // r11d
  __int64 *v14; // rdi
  unsigned int v15; // r8d
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 *v19; // r15
  int v20; // eax
  int v21; // r8d
  __int64 v22; // rsi
  unsigned int v23; // eax
  int v24; // ecx
  __int64 v25; // rdx
  int v26; // r10d
  __int64 *v27; // r9
  int v28; // eax
  int v29; // eax
  int v30; // eax
  __int64 v31; // r8
  int v32; // r10d
  unsigned int v33; // edx
  __int64 v34; // rsi
  unsigned int v35; // [rsp+4h] [rbp-DCh]
  __int64 v36; // [rsp+8h] [rbp-D8h]
  __int64 v37; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v38; // [rsp+18h] [rbp-C8h]
  __int64 v39; // [rsp+20h] [rbp-C0h]
  __int64 v40; // [rsp+28h] [rbp-B8h]
  __int64 v41; // [rsp+30h] [rbp-B0h]
  __int64 *v42; // [rsp+38h] [rbp-A8h]
  __int64 v43; // [rsp+40h] [rbp-A0h]
  __int64 v44; // [rsp+48h] [rbp-98h]
  __int64 v45; // [rsp+50h] [rbp-90h]
  __int64 v46; // [rsp+58h] [rbp-88h]
  __int64 v47; // [rsp+60h] [rbp-80h]
  __int64 v48; // [rsp+68h] [rbp-78h]
  __int64 v49; // [rsp+70h] [rbp-70h]
  __int64 v50; // [rsp+78h] [rbp-68h]
  __int64 v51; // [rsp+80h] [rbp-60h]
  __int64 v52; // [rsp+88h] [rbp-58h]
  __int64 *v53; // [rsp+90h] [rbp-50h]
  __int64 *v54; // [rsp+98h] [rbp-48h]
  __int64 v55; // [rsp+A0h] [rbp-40h]
  char v56; // [rsp+A8h] [rbp-38h]

  v3 = a1 + 8;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  v36 = a1 + 72;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  sub_BD22F0((__int64)&v37, a2, 0);
  v4 = v54;
  for ( i = v53; v4 != i; ++i )
  {
    while ( 1 )
    {
      v6 = *i;
      if ( (*(_BYTE *)(*i + 9) & 1) != 0 )
        break;
      sub_10641A0(v3, v6);
      if ( v4 == ++i )
        goto LABEL_6;
    }
    sub_1063E40(v3, v6);
  }
LABEL_6:
  v7 = v42;
  v8 = &v42[(unsigned int)v44];
  if ( (_DWORD)v43 && v42 != v8 )
  {
    while ( *v7 == -4096 || *v7 == -8192 )
    {
      if ( ++v7 == v8 )
        goto LABEL_7;
    }
    if ( v8 != v7 )
    {
      v10 = *(_DWORD *)(a1 + 96);
      v11 = *v7;
      if ( !v10 )
        goto LABEL_27;
LABEL_17:
      v12 = *(_QWORD *)(a1 + 80);
      v13 = 1;
      v14 = 0;
      v15 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v16 = (__int64 *)(v12 + 16LL * v15);
      v17 = *v16;
      if ( v11 == *v16 )
        goto LABEL_18;
      while ( v17 != -4096 )
      {
        if ( !v14 && v17 == -8192 )
          v14 = v16;
        v15 = (v10 - 1) & (v13 + v15);
        v16 = (__int64 *)(v12 + 16LL * v15);
        v17 = *v16;
        if ( v11 == *v16 )
        {
LABEL_18:
          v18 = v16[1];
          v19 = v16 + 1;
          if ( v18 )
            sub_B91220((__int64)(v16 + 1), v18);
          goto LABEL_20;
        }
        ++v13;
      }
      if ( !v14 )
        v14 = v16;
      v28 = *(_DWORD *)(a1 + 88);
      ++*(_QWORD *)(a1 + 72);
      v24 = v28 + 1;
      if ( 4 * (v28 + 1) >= 3 * v10 )
      {
        while ( 1 )
        {
          sub_FC7EB0(v36, 2 * v10);
          v20 = *(_DWORD *)(a1 + 96);
          if ( !v20 )
            goto LABEL_61;
          v21 = v20 - 1;
          v22 = *(_QWORD *)(a1 + 80);
          v23 = (v20 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v24 = *(_DWORD *)(a1 + 88) + 1;
          v14 = (__int64 *)(v22 + 16LL * v23);
          v25 = *v14;
          if ( v11 != *v14 )
            break;
LABEL_45:
          *(_DWORD *)(a1 + 88) = v24;
          if ( *v14 != -4096 )
            --*(_DWORD *)(a1 + 92);
          *v14 = v11;
          v19 = v14 + 1;
          v14[1] = 0;
LABEL_20:
          *v19 = v11;
          if ( v11 )
            sub_B96E90((__int64)v19, v11, 1);
          do
          {
            if ( ++v7 == v8 )
              goto LABEL_7;
          }
          while ( *v7 == -8192 || *v7 == -4096 );
          if ( v7 == v8 )
            goto LABEL_7;
          v10 = *(_DWORD *)(a1 + 96);
          v11 = *v7;
          if ( v10 )
            goto LABEL_17;
LABEL_27:
          ++*(_QWORD *)(a1 + 72);
        }
        v26 = 1;
        v27 = 0;
        while ( v25 != -4096 )
        {
          if ( v25 == -8192 && !v27 )
            v27 = v14;
          v23 = v21 & (v26 + v23);
          v14 = (__int64 *)(v22 + 16LL * v23);
          v25 = *v14;
          if ( v11 == *v14 )
            goto LABEL_45;
          ++v26;
        }
      }
      else
      {
        if ( v10 - *(_DWORD *)(a1 + 92) - v24 > v10 >> 3 )
          goto LABEL_45;
        v35 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
        sub_FC7EB0(v36, v10);
        v29 = *(_DWORD *)(a1 + 96);
        if ( !v29 )
        {
LABEL_61:
          ++*(_DWORD *)(a1 + 88);
          BUG();
        }
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a1 + 80);
        v27 = 0;
        v32 = 1;
        v33 = v30 & v35;
        v24 = *(_DWORD *)(a1 + 88) + 1;
        v14 = (__int64 *)(v31 + 16LL * (v30 & v35));
        v34 = *v14;
        if ( v11 == *v14 )
          goto LABEL_45;
        while ( v34 != -4096 )
        {
          if ( v34 == -8192 && !v27 )
            v27 = v14;
          v33 = v30 & (v32 + v33);
          v14 = (__int64 *)(v31 + 16LL * v33);
          v34 = *v14;
          if ( v11 == *v14 )
            goto LABEL_45;
          ++v32;
        }
      }
      if ( v27 )
        v14 = v27;
      goto LABEL_45;
    }
  }
LABEL_7:
  if ( v53 )
    j_j___libc_free_0(v53, v55 - (_QWORD)v53);
  sub_C7D6A0(v50, 8LL * (unsigned int)v52, 8);
  sub_C7D6A0(v46, 8LL * (unsigned int)v48, 8);
  sub_C7D6A0((__int64)v42, 8LL * (unsigned int)v44, 8);
  return sub_C7D6A0(v38, 8LL * (unsigned int)v40, 8);
}
