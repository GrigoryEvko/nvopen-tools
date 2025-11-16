// Function: sub_141C430
// Address: 0x141c430
//
__int64 __fastcall sub_141C430(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned int v6; // esi
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 result; // rax
  int v12; // r10d
  __int64 *v13; // r8
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  unsigned __int64 v17; // r13
  __int64 v18; // r8
  bool v19; // al
  _QWORD *v20; // rdx
  unsigned __int8 v21; // al
  unsigned __int8 v22; // dl
  __int64 *v23; // rax
  __int64 v24; // rax
  char v25; // al
  __int64 v26; // r14
  unsigned __int64 v27; // rdx
  unsigned __int8 v28; // al
  __int64 v29; // rsi
  unsigned __int8 v30; // al
  int v31; // eax
  __int64 v32; // rdi
  unsigned int v33; // eax
  __int64 v34; // rsi
  int v35; // r10d
  __int64 *v36; // r8
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  int v40; // r8d
  unsigned int v41; // r13d
  __int64 *v42; // rdi
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-78h]
  __int64 v46; // [rsp+10h] [rbp-70h]
  __int64 *v47; // [rsp+18h] [rbp-68h]
  __m128i v48; // [rsp+20h] [rbp-60h] BYREF
  __int64 v49; // [rsp+30h] [rbp-50h]
  __int64 v50; // [rsp+38h] [rbp-48h]
  __int64 v51; // [rsp+40h] [rbp-40h]

  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_39;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( *v9 != a2 )
  {
    v12 = 1;
    v13 = 0;
    while ( v10 != -8 )
    {
      if ( v10 == -16 && !v13 )
        v13 = v9;
      v8 = (v6 - 1) & (v12 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( *v9 == a2 )
        goto LABEL_3;
      ++v12;
    }
    v14 = *(_DWORD *)(a1 + 16);
    if ( v13 )
      v9 = v13;
    ++*(_QWORD *)a1;
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v6 )
    {
      v16 = v6 >> 3;
      if ( v6 - *(_DWORD *)(a1 + 20) - v15 > (unsigned int)v16 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 16) = v15;
        if ( *v9 != -8 )
          --*(_DWORD *)(a1 + 20);
        *v9 = a2;
        v17 = a2;
        v9[1] = 0;
        v47 = v9 + 1;
LABEL_14:
        v18 = *(_QWORD *)(a2 + 40);
        if ( *(_BYTE *)(a2 + 16) == 54 )
        {
          v46 = *(_QWORD *)(a2 + 40);
          v19 = sub_14152B0(a1 + 832, a2);
          v18 = v46;
          if ( v19 )
          {
            v9[1] = 0x2000000000000003LL;
            return 0x2000000000000003LL;
          }
        }
        if ( *(_QWORD *)(v18 + 48) == a2 + 24 )
        {
          v24 = *(_QWORD *)(*(_QWORD *)(v18 + 56) + 80LL);
          if ( v24 && v18 == v24 - 24 )
            v9[1] = 0x4000000000000003LL;
          else
            v9[1] = 0x2000000000000003LL;
          return v9[1];
        }
        v20 = *(_QWORD **)(a1 + 272);
        v45 = v18;
        v48.m128i_i64[0] = 0;
        v48.m128i_i64[1] = -1;
        v49 = 0;
        v50 = 0;
        v51 = 0;
        v21 = sub_14113F0(a2, &v48, v20, v16, v18);
        if ( v48.m128i_i64[0] )
        {
          v22 = ((v21 >> 1) ^ 1) & 1;
          if ( *(_BYTE *)(a2 + 16) == 78 )
          {
            v43 = *(_QWORD *)(a2 - 24);
            if ( !*(_BYTE *)(v43 + 16) && (*(_BYTE *)(v43 + 33) & 0x20) != 0 )
              v22 |= *(_DWORD *)(v43 + 36) == 117;
          }
          *v47 = sub_141C340(a1, &v48, v22, (_QWORD *)(v17 + 24), v45, a2, 0, a3);
          goto LABEL_21;
        }
        v25 = *(_BYTE *)(a2 + 16);
        if ( v25 == 78 )
        {
          v26 = a2 | 4;
          v27 = a2 & 0xFFFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( v25 != 29 )
          {
            v9[1] = 0x6000000000000003LL;
LABEL_21:
            if ( ((_DWORD)v9[1] & 7u) <= 2 )
            {
              v48.m128i_i64[0] = v9[1] & 0xFFFFFFFFFFFFFFF8LL;
              if ( v48.m128i_i64[0] )
              {
                v23 = sub_1417C70(a1 + 192, v48.m128i_i64);
                sub_1412190((__int64)(v23 + 1), a2);
              }
            }
            return v9[1];
          }
          v26 = a2 & 0xFFFFFFFFFFFFFFFBLL;
          v27 = a2 & 0xFFFFFFFFFFFFFFF8LL;
        }
        v28 = *(_BYTE *)(v27 + 16);
        v29 = 0;
        if ( v28 > 0x17u )
        {
          if ( v28 == 78 )
          {
            v29 = v27 | 4;
          }
          else if ( v28 == 29 )
          {
            v29 = v27;
          }
        }
        v30 = sub_134CC90(*(_QWORD *)(a1 + 256), v29);
        *v47 = sub_1412B20(a1, v26, ((v30 >> 1) ^ 1) & 1, (_QWORD *)(v17 + 24), v45);
        goto LABEL_21;
      }
      sub_14178C0(a1, v6);
      v37 = *(_DWORD *)(a1 + 24);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a1 + 8);
        v40 = 1;
        v41 = v38 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = *(_DWORD *)(a1 + 16) + 1;
        v42 = 0;
        v9 = (__int64 *)(v39 + 16LL * v41);
        v16 = *v9;
        if ( *v9 != a2 )
        {
          while ( v16 != -8 )
          {
            if ( !v42 && v16 == -16 )
              v42 = v9;
            v41 = v38 & (v40 + v41);
            v9 = (__int64 *)(v39 + 16LL * v41);
            v16 = *v9;
            if ( *v9 == a2 )
              goto LABEL_11;
            ++v40;
          }
          if ( v42 )
            v9 = v42;
        }
        goto LABEL_11;
      }
LABEL_75:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_39:
    sub_14178C0(a1, 2 * v6);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v16 = (unsigned int)(v31 - 1);
      v32 = *(_QWORD *)(a1 + 8);
      v33 = v16 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v9 = (__int64 *)(v32 + 16LL * v33);
      v34 = *v9;
      if ( *v9 != a2 )
      {
        v35 = 1;
        v36 = 0;
        while ( v34 != -8 )
        {
          if ( !v36 && v34 == -16 )
            v36 = v9;
          v33 = v16 & (v35 + v33);
          v9 = (__int64 *)(v32 + 16LL * v33);
          v34 = *v9;
          if ( *v9 == a2 )
            goto LABEL_11;
          ++v35;
        }
        if ( v36 )
          v9 = v36;
      }
      goto LABEL_11;
    }
    goto LABEL_75;
  }
LABEL_3:
  result = v9[1];
  if ( (result & 7) == 0 )
  {
    v16 = (__int64)(v9 + 1);
    v44 = result & 0xFFFFFFFFFFFFFFF8LL;
    v47 = v9 + 1;
    v17 = v44;
    if ( v44 )
      sub_1411E70(a1 + 192, v44, a2);
    else
      v17 = a2;
    goto LABEL_14;
  }
  return result;
}
