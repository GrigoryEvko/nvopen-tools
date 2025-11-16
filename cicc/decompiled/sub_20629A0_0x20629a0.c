// Function: sub_20629A0
// Address: 0x20629a0
//
__int64 __fastcall sub_20629A0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // rsi
  char v5; // dl
  __int64 v6; // rax
  unsigned int v7; // r8d
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 result; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r12
  int v17; // eax
  __int64 v18; // rsi
  int v19; // ecx
  int v20; // edi
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // r13
  __int64 v25; // rsi
  int v26; // ecx
  int v27; // edi
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 *v30; // r13
  __int64 *v31; // r12
  __int64 v32; // r14
  int v33; // edx
  int v34; // edi
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rcx
  int v40; // eax
  __int64 v41; // rdi
  int v42; // edx
  __int64 v43; // rsi
  int v44; // r8d
  unsigned int v45; // esi
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // r12
  int v51; // edx
  int v52; // r12d
  __int64 v53; // r8
  __int64 v54; // [rsp+8h] [rbp-148h]
  __int64 v55; // [rsp+10h] [rbp-140h]
  unsigned __int64 v56; // [rsp+18h] [rbp-138h]
  int v57; // [rsp+24h] [rbp-12Ch]
  __int64 *v60; // [rsp+38h] [rbp-118h]
  __int64 v61; // [rsp+38h] [rbp-118h]
  __int64 v63; // [rsp+48h] [rbp-108h]
  __int64 v64; // [rsp+50h] [rbp-100h] BYREF
  int v65; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v66; // [rsp+60h] [rbp-F0h] BYREF
  __m128i v67; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v68; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+98h] [rbp-B8h]
  __int64 v70; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int v71; // [rsp+A8h] [rbp-A8h]
  char v72; // [rsp+120h] [rbp-30h] BYREF

  v3 = &v70;
  v68 = 0;
  v69 = 1;
  do
  {
    *v3 = -8;
    v3 += 2;
  }
  while ( v3 != (__int64 *)&v72 );
  v4 = *a2;
  v5 = v69 & 1;
  v6 = *(_QWORD *)(*a2 + 96);
  v57 = v6;
  v7 = 2 * v6;
  if ( 2 * (_DWORD)v6 )
  {
    ++v68;
    v8 = (8 * (int)v6 / 3u + 1) | ((unsigned __int64)(8 * (int)v6 / 3u + 1) >> 1);
    v9 = (((v8 >> 2) | v8) >> 4) | (v8 >> 2) | v8;
    v7 = ((((v9 >> 8) | v9) >> 16) | (v9 >> 8) | v9) + 1;
    v10 = 8;
    if ( v5 )
      goto LABEL_6;
  }
  else
  {
    ++v68;
    if ( v5 )
      goto LABEL_8;
  }
  v10 = v71;
LABEL_6:
  if ( v7 > v10 )
  {
    sub_2046A80((__int64)&v68, v7);
    v4 = *a2;
  }
LABEL_8:
  v11 = *(_QWORD *)(v4 + 80);
  if ( !v11 )
    BUG();
  v12 = *(_QWORD *)(v11 + 24);
  result = v11 + 16;
  v63 = result;
  if ( v12 != result )
  {
    while ( 1 )
    {
      if ( !v12 )
        BUG();
      result = *(unsigned __int8 *)(v12 - 8);
      if ( (_BYTE)result != 55 )
      {
        if ( (unsigned int)(unsigned __int8)result - 60 > 0xC )
        {
          if ( (_BYTE)result != 78
            || (v37 = *(_QWORD *)(v12 - 48), *(_BYTE *)(v37 + 16))
            || (*(_BYTE *)(v37 + 33) & 0x20) == 0
            || (result = (unsigned int)(*(_DWORD *)(v37 + 36) - 35), (unsigned int)result > 3) )
          {
            result = 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF);
            if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
            {
              v30 = *(__int64 **)(v12 - 32);
              v31 = (__int64 *)((char *)v30 + result);
            }
            else
            {
              v31 = (__int64 *)(v12 - 24);
              v30 = (__int64 *)(v12 - 24 - result);
            }
            for ( ; v31 != v30; v30 += 3 )
            {
              if ( *v30 )
              {
                result = sub_1649C60(*v30);
                v32 = result;
                if ( *(_BYTE *)(result + 16) == 53 )
                {
                  result = sub_15F8F00(result);
                  if ( (_BYTE)result )
                  {
                    result = *((unsigned int *)a2 + 90);
                    if ( (_DWORD)result )
                    {
                      v33 = result - 1;
                      v34 = 1;
                      v35 = a2[43];
                      result = ((_DWORD)result - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
                      v36 = *(_QWORD *)(v35 + 16 * result);
                      if ( v32 == v36 )
                      {
LABEL_39:
                        v64 = v32;
                        v65 = 0;
                        sub_2047830((__int64)&v66, (__int64)&v68, &v64, &v65);
                        result = v67.m128i_i64[1];
                        *(_DWORD *)(v67.m128i_i64[1] + 8) = 1;
                      }
                      else
                      {
                        while ( v36 != -8 )
                        {
                          result = v33 & (unsigned int)(v34 + result);
                          v36 = *(_QWORD *)(v35 + 16LL * (unsigned int)result);
                          if ( v32 == v36 )
                            goto LABEL_39;
                          ++v34;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        goto LABEL_29;
      }
      v14 = *(_QWORD *)(v12 - 72);
      if ( v14 )
      {
        v15 = sub_1649C60(v14);
        v16 = v15;
        if ( *(_BYTE *)(v15 + 16) == 53 )
        {
          if ( (unsigned __int8)sub_15F8F00(v15) )
          {
            v17 = *((_DWORD *)a2 + 90);
            if ( v17 )
            {
              v18 = a2[43];
              v19 = v17 - 1;
              v20 = 1;
              v21 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
              v22 = *(_QWORD *)(v18 + 16LL * v21);
              if ( v16 == v22 )
              {
LABEL_17:
                v64 = v16;
                v65 = 0;
                sub_2047830((__int64)&v66, (__int64)&v68, &v64, &v65);
                *(_DWORD *)(v67.m128i_i64[1] + 8) = 1;
              }
              else
              {
                while ( v22 != -8 )
                {
                  v21 = v19 & (v20 + v21);
                  v22 = *(_QWORD *)(v18 + 16LL * v21);
                  if ( v16 == v22 )
                    goto LABEL_17;
                  ++v20;
                }
              }
            }
          }
        }
      }
      result = sub_1649C60(*(_QWORD *)(v12 - 48));
      v23 = result;
      if ( result )
      {
        result = sub_1649C60(result);
        v24 = result;
        if ( *(_BYTE *)(result + 16) == 53 )
        {
          result = sub_15F8F00(result);
          if ( (_BYTE)result )
          {
            result = *((unsigned int *)a2 + 90);
            if ( (_DWORD)result )
            {
              v25 = a2[43];
              v26 = result - 1;
              v27 = 1;
              result = ((_DWORD)result - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v28 = *(_QWORD *)(v25 + 16 * result);
              if ( v24 != v28 )
              {
                while ( v28 != -8 )
                {
                  result = v26 & (unsigned int)(v27 + result);
                  v28 = *(_QWORD *)(v25 + 16LL * (unsigned int)result);
                  if ( v24 == v28 )
                    goto LABEL_23;
                  ++v27;
                }
                goto LABEL_29;
              }
LABEL_23:
              v64 = v24;
              v65 = 0;
              sub_2047830((__int64)&v66, (__int64)&v68, &v64, &v65);
              v29 = v67.m128i_i64[1];
              result = *(unsigned int *)(v67.m128i_i64[1] + 8);
              if ( !(_DWORD)result )
                break;
            }
          }
        }
      }
LABEL_29:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v63 == v12 )
        goto LABEL_30;
    }
    result = sub_1649C60(*(_QWORD *)(v12 - 72));
    if ( *(_BYTE *)(result + 16) != 17 )
      goto LABEL_28;
    v60 = (__int64 *)result;
    result = sub_15E0470(result);
    if ( (_BYTE)result )
      goto LABEL_28;
    result = sub_15E0450((__int64)v60);
    if ( (_BYTE)result )
      goto LABEL_28;
    result = sub_1642FB0(*v60);
    if ( (_BYTE)result
      || (v54 = (__int64)v60,
          v61 = sub_127FA20(a1, *v60),
          v55 = *(_QWORD *)(v23 + 56),
          v56 = (unsigned int)sub_15A9FE0(a1, v55),
          v38 = sub_127FA20(a1, v55),
          v39 = v54,
          result = v56 * ((v56 + ((unsigned __int64)(v38 + 7) >> 3) - 1) / v56),
          (unsigned __int64)(v61 + 7) >> 3 != result) )
    {
LABEL_28:
      *(_DWORD *)(v29 + 8) = 1;
      goto LABEL_29;
    }
    v40 = *(_DWORD *)(a3 + 24);
    if ( v40 )
    {
      v41 = *(_QWORD *)(a3 + 8);
      v42 = v40 - 1;
      result = (v40 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v43 = *(_QWORD *)(v41 + 24 * result);
      if ( v54 == v43 )
        goto LABEL_28;
      v44 = 1;
      while ( v43 != -8 )
      {
        result = v42 & (unsigned int)(v44 + result);
        v43 = *(_QWORD *)(v41 + 24LL * (unsigned int)result);
        if ( v54 == v43 )
          goto LABEL_28;
        ++v44;
      }
    }
    *(_DWORD *)(v29 + 8) = 2;
    v67.m128i_i64[1] = v12 - 24;
    v66 = v54;
    v45 = *(_DWORD *)(a3 + 24);
    v67.m128i_i64[0] = v23;
    if ( v45 )
    {
      v46 = *(_QWORD *)(a3 + 8);
      LODWORD(v47) = (v45 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v48 = v46 + 24LL * (unsigned int)v47;
      v49 = *(_QWORD *)v48;
      if ( v54 == *(_QWORD *)v48 )
        goto LABEL_73;
      v52 = 1;
      v53 = 0;
      while ( v49 != -8 )
      {
        if ( v49 == -16 && !v53 )
          v53 = v48;
        v47 = (v45 - 1) & ((_DWORD)v47 + v52);
        v48 = v46 + 24 * v47;
        v49 = *(_QWORD *)v48;
        if ( v54 == *(_QWORD *)v48 )
          goto LABEL_73;
        ++v52;
      }
      if ( v53 )
        v48 = v53;
      ++*(_QWORD *)a3;
      v51 = *(_DWORD *)(a3 + 16) + 1;
      if ( 4 * v51 < 3 * v45 )
      {
        if ( v45 - *(_DWORD *)(a3 + 20) - v51 > v45 >> 3 )
          goto LABEL_78;
        v50 = a3;
LABEL_77:
        sub_20627D0(v50, v45);
        sub_205ACC0(v50, &v66, &v64);
        v48 = v64;
        v39 = v66;
        v51 = *(_DWORD *)(v50 + 16) + 1;
LABEL_78:
        *(_DWORD *)(a3 + 16) = v51;
        if ( *(_QWORD *)v48 != -8 )
          --*(_DWORD *)(a3 + 20);
        *(_QWORD *)v48 = v39;
        *(__m128i *)(v48 + 8) = _mm_loadu_si128(&v67);
LABEL_73:
        result = a3;
        if ( v57 == *(_DWORD *)(a3 + 16) )
          goto LABEL_30;
        goto LABEL_29;
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    v50 = a3;
    v45 *= 2;
    goto LABEL_77;
  }
LABEL_30:
  if ( (v69 & 1) == 0 )
    return j___libc_free_0(v70);
  return result;
}
