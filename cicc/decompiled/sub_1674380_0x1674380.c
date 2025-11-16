// Function: sub_1674380
// Address: 0x1674380
//
__int64 __fastcall sub_1674380(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r13
  __int64 *v4; // r12
  __int64 *i; // rbx
  __int64 v6; // rsi
  __int64 *v7; // rbx
  __int64 *v8; // r12
  unsigned int v10; // esi
  __int64 v11; // r14
  __int64 v12; // r8
  unsigned int v13; // edi
  __int64 *v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdi
  int v18; // r11d
  __int64 *v19; // r10
  int v20; // ecx
  int v21; // edi
  int v22; // eax
  int v23; // r8d
  __int64 v24; // rsi
  unsigned int v25; // eax
  __int64 v26; // rdx
  int v27; // r10d
  __int64 *v28; // r9
  int v29; // edx
  int v30; // edx
  __int64 v31; // r8
  int v32; // r10d
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // [rsp+0h] [rbp-C0h]
  unsigned int v36; // [rsp+8h] [rbp-B8h]
  __int64 v37; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+18h] [rbp-A8h]
  __int64 v39; // [rsp+20h] [rbp-A0h]
  __int64 v40; // [rsp+28h] [rbp-98h]
  __int64 v41; // [rsp+30h] [rbp-90h]
  __int64 *v42; // [rsp+38h] [rbp-88h]
  __int64 v43; // [rsp+40h] [rbp-80h]
  __int64 v44; // [rsp+48h] [rbp-78h]
  __int64 v45; // [rsp+50h] [rbp-70h]
  __int64 v46; // [rsp+58h] [rbp-68h]
  __int64 v47; // [rsp+60h] [rbp-60h]
  __int64 v48; // [rsp+68h] [rbp-58h]
  __int64 *v49; // [rsp+70h] [rbp-50h]
  __int64 *v50; // [rsp+78h] [rbp-48h]
  __int64 v51; // [rsp+80h] [rbp-40h]
  char v52; // [rsp+88h] [rbp-38h]

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
  v35 = a1 + 72;
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
  sub_1648080((__int64)&v37, a2, 0);
  v4 = v50;
  for ( i = v49; v4 != i; ++i )
  {
    while ( 1 )
    {
      v6 = *i;
      if ( (*(_BYTE *)(*i + 9) & 1) != 0 )
        break;
      sub_1674160(v3, v6);
      if ( v4 == ++i )
        goto LABEL_6;
    }
    sub_1673BC0(v3, v6);
  }
LABEL_6:
  v7 = v42;
  v8 = &v42[(unsigned int)v44];
  if ( (_DWORD)v43 && v42 != v8 )
  {
    while ( *v7 == -8 || *v7 == -16 )
    {
      if ( ++v7 == v8 )
        goto LABEL_7;
    }
LABEL_15:
    if ( v7 != v8 )
    {
      v10 = *(_DWORD *)(a1 + 96);
      v11 = *v7;
      if ( v10 )
      {
        v12 = *(_QWORD *)(a1 + 80);
        v13 = (v10 - 1) & (((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9));
        v14 = (__int64 *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v11 == *v14 )
        {
LABEL_18:
          v16 = v14[1];
          v17 = (__int64)(v14 + 1);
          if ( v16 )
          {
            sub_161E7C0(v17, v16);
            v17 = (__int64)(v14 + 1);
          }
          goto LABEL_20;
        }
        v18 = 1;
        v19 = 0;
        while ( v15 != -4 )
        {
          if ( !v19 && v15 == -8 )
            v19 = v14;
          v13 = (v10 - 1) & (v18 + v13);
          v14 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( v11 == *v14 )
            goto LABEL_18;
          ++v18;
        }
        v20 = *(_DWORD *)(a1 + 88);
        if ( v19 )
          v14 = v19;
        ++*(_QWORD *)(a1 + 72);
        v21 = v20 + 1;
        if ( 4 * (v20 + 1) < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a1 + 92) - v21 > v10 >> 3 )
          {
LABEL_32:
            *(_DWORD *)(a1 + 88) = v21;
            if ( *v14 != -4 )
              --*(_DWORD *)(a1 + 92);
            *v14 = v11;
            v17 = (__int64)(v14 + 1);
            v14[1] = 0;
LABEL_20:
            v14[1] = v11;
            if ( v11 )
              sub_1623A60(v17, v11, 2);
            while ( 1 )
            {
              if ( ++v7 == v8 )
                goto LABEL_7;
              if ( *v7 != -16 && *v7 != -8 )
                goto LABEL_15;
            }
          }
          v36 = ((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9);
          sub_1671930(v35, v10);
          v29 = *(_DWORD *)(a1 + 96);
          if ( v29 )
          {
            v30 = v29 - 1;
            v31 = *(_QWORD *)(a1 + 80);
            v28 = 0;
            v32 = 1;
            v33 = v30 & v36;
            v21 = *(_DWORD *)(a1 + 88) + 1;
            v14 = (__int64 *)(v31 + 16LL * (v30 & v36));
            v34 = *v14;
            if ( v11 == *v14 )
              goto LABEL_32;
            while ( v34 != -4 )
            {
              if ( v34 == -8 && !v28 )
                v28 = v14;
              v33 = v30 & (v32 + v33);
              v14 = (__int64 *)(v31 + 16LL * v33);
              v34 = *v14;
              if ( v11 == *v14 )
                goto LABEL_32;
              ++v32;
            }
            goto LABEL_40;
          }
          goto LABEL_61;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 72);
      }
      sub_1671930(v35, 2 * v10);
      v22 = *(_DWORD *)(a1 + 96);
      if ( v22 )
      {
        v23 = v22 - 1;
        v24 = *(_QWORD *)(a1 + 80);
        v25 = (v22 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v21 = *(_DWORD *)(a1 + 88) + 1;
        v14 = (__int64 *)(v24 + 16LL * v25);
        v26 = *v14;
        if ( v11 == *v14 )
          goto LABEL_32;
        v27 = 1;
        v28 = 0;
        while ( v26 != -4 )
        {
          if ( v26 == -8 && !v28 )
            v28 = v14;
          v25 = v23 & (v27 + v25);
          v14 = (__int64 *)(v24 + 16LL * v25);
          v26 = *v14;
          if ( v11 == *v14 )
            goto LABEL_32;
          ++v27;
        }
LABEL_40:
        if ( v28 )
          v14 = v28;
        goto LABEL_32;
      }
LABEL_61:
      ++*(_DWORD *)(a1 + 88);
      BUG();
    }
  }
LABEL_7:
  if ( v49 )
    j_j___libc_free_0(v49, v51 - (_QWORD)v49);
  j___libc_free_0(v46);
  j___libc_free_0(v42);
  return j___libc_free_0(v38);
}
