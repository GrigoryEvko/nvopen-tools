// Function: sub_1FE92A0
// Address: 0x1fe92a0
//
__int64 __fastcall sub_1FE92A0(__int64 *a1, unsigned __int64 a2, __int64 a3, char a4, unsigned __int8 a5)
{
  __int64 *v5; // r15
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rdi
  size_t v11; // r12
  _QWORD *v12; // rax
  __int64 v13; // r8
  int v14; // r9d
  __int64 v15; // r12
  _QWORD *v16; // rax
  __int64 v17; // r14
  int v18; // eax
  __int64 v19; // rax
  unsigned __int64 v20; // r10
  unsigned __int64 v21; // r11
  __int64 v22; // r13
  __int64 *v23; // r14
  unsigned int v24; // r15d
  __int64 v25; // rdx
  unsigned __int64 *v26; // rax
  unsigned __int64 *v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // r13
  int v30; // eax
  __int64 v31; // rax
  _QWORD *v32; // r13
  __int64 *v33; // r12
  __int64 v34; // rcx
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // r8
  int v38; // r9d
  __int64 result; // rax
  unsigned int i; // ecx
  __int64 *v41; // rdx
  __int64 v42; // r13
  unsigned int v43; // ecx
  int v44; // eax
  int v45; // ecx
  __int64 v46; // r8
  int v47; // edi
  __int64 v48; // rsi
  unsigned int v49; // edx
  unsigned int v50; // edx
  int v51; // r10d
  int v52; // ecx
  int v53; // edi
  int v54; // eax
  int v55; // edx
  __int64 v56; // rdi
  int v57; // ecx
  unsigned int j; // r12d
  unsigned int v59; // r12d
  int v60; // esi
  int v61; // r9d
  unsigned __int64 v62; // [rsp+0h] [rbp-A0h]
  int v63; // [rsp+8h] [rbp-98h]
  int v64; // [rsp+14h] [rbp-8Ch]
  _QWORD *v65; // [rsp+18h] [rbp-88h]
  unsigned __int8 v66; // [rsp+20h] [rbp-80h]
  __int64 v68; // [rsp+30h] [rbp-70h]
  int v69; // [rsp+30h] [rbp-70h]
  __m128i v71; // [rsp+40h] [rbp-60h] BYREF
  __int64 v72; // [rsp+50h] [rbp-50h]
  __int64 v73; // [rsp+58h] [rbp-48h]
  __int64 v74; // [rsp+60h] [rbp-40h]

  v5 = a1;
  v8 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 88LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = a1[3];
  v11 = v5[1];
  v65 = *(_QWORD **)(*(_QWORD *)(v10 + 256) + 8LL * (unsigned int)v9);
  v12 = sub_1F4AAF0(v10, v65);
  v64 = sub_1E6B9A0(v11, (__int64)v12, (unsigned __int8 *)byte_3F871B3, 0, v13, v14);
  v68 = *v5;
  v15 = *(_QWORD *)(v5[2] + 8) + 896LL;
  v16 = sub_1E0B640(*v5, v15, (__int64 *)(a2 + 72), 0);
  v71.m128i_i64[0] = 0x10000000;
  v17 = (__int64)v16;
  v72 = 0;
  v73 = 0;
  v71.m128i_i32[2] = v64;
  v74 = 0;
  sub_1E1A9C0((__int64)v16, v68, &v71);
  v71.m128i_i64[1] = v17;
  v18 = *(_DWORD *)(a2 + 56);
  v71.m128i_i64[0] = v68;
  v69 = v18;
  if ( v18 != 1 )
  {
    v19 = *(_QWORD *)(a2 + 32);
    v20 = *(_QWORD *)(v19 + 40);
    v21 = *(_QWORD *)(v19 + 48);
    LOBYTE(v19) = a4;
    v22 = 1;
    v23 = v5;
    v66 = v19;
    while ( 1 )
    {
      v24 = v22 + 1;
      sub_1FE6BA0(v23, v71.m128i_i64, v20, v21, v22 + 1, v15, a3, 0, v66, a5);
      if ( (_DWORD)v22 + 1 == v69 )
        break;
      v25 = *(_QWORD *)(a2 + 32);
      v26 = (unsigned __int64 *)(v25 + 40LL * v24);
      v20 = *v26;
      v21 = v26[1];
      if ( (v24 & 1) == 0 )
      {
        v27 = (unsigned __int64 *)(v25 + 40 * v22);
        if ( *(_WORD *)(*v27 + 24) != 8 || *(int *)(*v27 + 84) <= 0 )
        {
          v28 = *(_QWORD *)(*v26 + 88);
          v29 = *(_QWORD **)(v28 + 24);
          if ( *(_DWORD *)(v28 + 32) > 0x40u )
            v29 = (_QWORD *)*v29;
          v62 = v20;
          v63 = v21;
          v30 = sub_1FE6610((size_t *)v23, *v27, v27[1], a3);
          v31 = (*(__int64 (__fastcall **)(__int64, _QWORD *, unsigned __int64, _QWORD))(*(_QWORD *)v23[3] + 96LL))(
                  v23[3],
                  v65,
                  *(_QWORD *)(*(_QWORD *)(v23[1] + 24) + 16LL * (v30 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                  (unsigned int)v29);
          v20 = v62;
          LODWORD(v21) = v63;
          v32 = (_QWORD *)v31;
          if ( v31 )
          {
            if ( v65 != (_QWORD *)v31 )
            {
              sub_1E693D0(v23[1], v64, v31);
              v65 = v32;
              v20 = v62;
              LODWORD(v21) = v63;
            }
          }
        }
      }
      v22 = v24;
    }
    v5 = v23;
    v17 = v71.m128i_i64[1];
  }
  v33 = (__int64 *)v5[6];
  sub_1DD5BA0((__int64 *)(v5[5] + 16), v17);
  v34 = *v33;
  v35 = *(_QWORD *)v17;
  *(_QWORD *)(v17 + 8) = v33;
  v34 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v17 = v34 | v35 & 7;
  *(_QWORD *)(v34 + 8) = v17;
  *v33 = *v33 & 7 | v17;
  v36 = *(_DWORD *)(a3 + 24);
  if ( !v36 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_24;
  }
  v37 = *(_QWORD *)(a3 + 8);
  v38 = 1;
  result = 0;
  for ( i = (v36 - 1) & ((a2 >> 9) ^ (a2 >> 4)); ; i = (v36 - 1) & v43 )
  {
    v41 = (__int64 *)(v37 + 24LL * i);
    v42 = *v41;
    if ( a2 != *v41 )
      break;
    if ( !*((_DWORD *)v41 + 2) )
      return result;
LABEL_20:
    v43 = v38 + i;
    ++v38;
  }
  if ( v42 )
    goto LABEL_20;
  v51 = *((_DWORD *)v41 + 2);
  if ( v51 != -1 )
  {
    if ( v51 == -2 && !result )
      result = v37 + 24LL * i;
    goto LABEL_20;
  }
  v53 = *(_DWORD *)(a3 + 16);
  if ( !result )
    result = v37 + 24LL * i;
  ++*(_QWORD *)a3;
  v52 = v53 + 1;
  if ( 4 * (v53 + 1) < 3 * v36 )
  {
    if ( v36 - *(_DWORD *)(a3 + 20) - v52 > v36 >> 3 )
      goto LABEL_36;
    sub_1FE7AA0(a3, v36);
    v54 = *(_DWORD *)(a3 + 24);
    if ( v54 )
    {
      v55 = v54 - 1;
      v57 = 1;
      for ( j = (v54 - 1) & ((a2 >> 9) ^ (a2 >> 4)); ; j = v55 & v59 )
      {
        v56 = *(_QWORD *)(a3 + 8);
        result = v56 + 24LL * j;
        if ( a2 == *(_QWORD *)result )
        {
          if ( !*(_DWORD *)(result + 8) )
            goto LABEL_35;
        }
        else if ( !*(_QWORD *)result )
        {
          v60 = *(_DWORD *)(result + 8);
          if ( v60 == -1 )
          {
            if ( v42 )
              result = v42;
            v52 = *(_DWORD *)(a3 + 16) + 1;
            goto LABEL_36;
          }
          if ( !v42 && v60 == -2 )
            v42 = v56 + 24LL * j;
        }
        v59 = v57 + j;
        ++v57;
      }
    }
LABEL_68:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
LABEL_24:
  sub_1FE7AA0(a3, 2 * v36);
  v44 = *(_DWORD *)(a3 + 24);
  if ( !v44 )
    goto LABEL_68;
  v45 = v44 - 1;
  v47 = 1;
  v48 = 0;
  v49 = (v44 - 1) & ((a2 >> 9) ^ (a2 >> 4));
  while ( 2 )
  {
    v46 = *(_QWORD *)(a3 + 8);
    result = v46 + 24LL * v49;
    if ( a2 == *(_QWORD *)result )
    {
      if ( !*(_DWORD *)(result + 8) )
      {
LABEL_35:
        v52 = *(_DWORD *)(a3 + 16) + 1;
        goto LABEL_36;
      }
      goto LABEL_28;
    }
    if ( *(_QWORD *)result )
    {
LABEL_28:
      v50 = v47 + v49;
      ++v47;
      v49 = v45 & v50;
      continue;
    }
    break;
  }
  v61 = *(_DWORD *)(result + 8);
  if ( v61 != -1 )
  {
    if ( v61 == -2 && !v48 )
      v48 = v46 + 24LL * v49;
    goto LABEL_28;
  }
  if ( v48 )
    result = v48;
  v52 = *(_DWORD *)(a3 + 16) + 1;
LABEL_36:
  *(_DWORD *)(a3 + 16) = v52;
  if ( *(_QWORD *)result || *(_DWORD *)(result + 8) != -1 )
    --*(_DWORD *)(a3 + 20);
  *(_DWORD *)(result + 8) = 0;
  *(_QWORD *)result = a2;
  *(_DWORD *)(result + 16) = v64;
  return result;
}
