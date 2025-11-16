// Function: sub_E5F0E0
// Address: 0xe5f0e0
//
__int64 *__fastcall sub_E5F0E0(__int64 a1)
{
  __int64 v1; // r15
  __int64 *v2; // r12
  __int64 *v3; // r14
  int v4; // r13d
  __int64 v5; // rbx
  int v6; // eax
  bool v7; // cc
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rsi
  __m128i *i; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __m128i *v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rax
  int v18; // edx
  int v19; // ecx
  __int64 *result; // rax
  __int64 *v21; // rax
  __int64 *v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rdi
  void (*v25)(); // rax
  __int64 v26; // rdi
  void (*v27)(); // rax
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // r15
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rbx
  __int64 *v35; // r12
  __int64 v36; // rdi
  __m128i v37; // xmm0
  __int64 *v38; // rdx
  __int64 v39; // rdi
  __int64 (*v40)(); // rax
  __int64 *v41; // [rsp+8h] [rbp-B8h]
  __int64 v42; // [rsp+10h] [rbp-B0h]
  __int64 *v43; // [rsp+18h] [rbp-A8h]
  __int64 v44; // [rsp+20h] [rbp-A0h]
  __int64 v45; // [rsp+28h] [rbp-98h]
  __int64 *v46; // [rsp+38h] [rbp-88h]
  __m128i v47; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v48; // [rsp+50h] [rbp-70h]
  int v49; // [rsp+58h] [rbp-68h]
  __m128i v50; // [rsp+60h] [rbp-60h] BYREF
  __m128i v51; // [rsp+70h] [rbp-50h] BYREF
  __m128i *v52; // [rsp+80h] [rbp-40h]
  int v53; // [rsp+88h] [rbp-38h]

  v1 = a1;
  v2 = *(__int64 **)(a1 + 40);
  v3 = &v2[*(unsigned int *)(a1 + 48)];
  if ( v2 != v3 )
  {
    v4 = 0;
    do
    {
      while ( 1 )
      {
        v5 = *v2;
        v6 = v4++;
        v7 = *(_DWORD *)(*v2 + 96) <= 1u;
        *(_DWORD *)(*v2 + 36) = v6;
        if ( !v7 )
        {
          sub_E81B30(&v50, 14, 0);
          v10 = *(_QWORD *)(v5 + 88);
          v11 = v10 + 24LL * *(unsigned int *)(v5 + 96);
          for ( i = &v50; v11 != v10; i = *(__m128i **)(v10 - 8) )
          {
            v13 = *(_QWORD *)(v10 + 8);
            v10 += 24;
            i->m128i_i64[0] = v13;
          }
          *(_DWORD *)(v5 + 96) = 0;
          v47.m128i_i32[0] = 0;
          v47.m128i_i64[1] = v50.m128i_i64[0];
          v14 = 0;
          v48 = i;
          if ( !*(_DWORD *)(v5 + 100) )
          {
            sub_C8D5F0(v5 + 88, (const void *)(v5 + 104), 1u, 0x18u, v8, v9);
            v14 = 24LL * *(unsigned int *)(v5 + 96);
          }
          v15 = (__m128i *)(*(_QWORD *)(v5 + 88) + v14);
          *v15 = _mm_loadu_si128(&v47);
          v15[1].m128i_i64[0] = (__int64)v48;
          v16 = *(_QWORD *)(v5 + 88);
          ++*(_DWORD *)(v5 + 96);
          *(_QWORD *)(v5 + 8) = v16 + 8;
          v17 = *(__int64 **)(v16 + 8);
          v18 = 0;
          if ( v17 )
            break;
        }
        if ( v3 == ++v2 )
          goto LABEL_12;
      }
      do
      {
        v19 = v18++;
        *((_DWORD *)v17 + 6) = v19;
        v17 = (__int64 *)*v17;
      }
      while ( v17 );
      ++v2;
    }
    while ( v3 != v2 );
LABEL_12:
    v1 = a1;
  }
  *(_BYTE *)(v1 + 32) = 1;
LABEL_14:
  if ( (unsigned __int8)sub_E5F060(v1) )
  {
    while ( 1 )
    {
      result = *(__int64 **)v1;
      if ( *(_BYTE *)(*(_QWORD *)v1 + 2376LL) )
        break;
      v21 = *(__int64 **)(v1 + 40);
      v22 = &v21[*(unsigned int *)(v1 + 48)];
      if ( v21 == v22 )
        goto LABEL_14;
      do
      {
        v23 = *v21++;
        *(_BYTE *)(v23 + 48) &= ~4u;
      }
      while ( v22 != v21 );
      if ( !(unsigned __int8)sub_E5F060(v1) )
        goto LABEL_19;
    }
  }
  else
  {
LABEL_19:
    v24 = *(_QWORD *)(v1 + 8);
    v25 = *(void (**)())(*(_QWORD *)v24 + 200LL);
    if ( v25 != nullsub_323 )
      ((void (__fastcall *)(__int64, __int64))v25)(v24, v1);
    v26 = *(_QWORD *)(v1 + 24);
    v27 = *(void (**)())(*(_QWORD *)v26 + 24LL);
    if ( v27 != nullsub_325 )
      ((void (__fastcall *)(__int64, __int64))v27)(v26, v1);
    result = *(__int64 **)(v1 + 40);
    v41 = &result[*(unsigned int *)(v1 + 48)];
    if ( v41 != result )
    {
      v43 = *(__int64 **)(v1 + 40);
      v28 = v1;
      do
      {
        v42 = *v43;
        if ( **(_QWORD **)(*v43 + 8) )
        {
          v29 = v28;
          v30 = **(_QWORD **)(*v43 + 8);
          v31 = v29;
          do
          {
            while ( 2 )
            {
              switch ( *(_BYTE *)(v30 + 28) )
              {
                case 0:
                  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v42 + 8LL))(v42) )
                    break;
                  if ( (*(_BYTE *)(v30 + 31) & 1) == 0 )
                    break;
                  v39 = *(_QWORD *)(v31 + 8);
                  v40 = *(__int64 (**)())(*(_QWORD *)v39 + 88LL);
                  if ( v40 == sub_E5B810 )
                    break;
                  ((void (__fastcall *)(__int64, __int64, __int64))v40)(v39, v31, v30);
                  v30 = *(_QWORD *)v30;
                  if ( v30 )
                    continue;
                  goto LABEL_33;
                case 1:
                  v32 = *(__int64 **)(v30 + 96);
                  v33 = *(unsigned int *)(v30 + 104);
                  goto LABEL_39;
                case 4:
                  v32 = *(__int64 **)(v30 + 72);
                  v33 = *(unsigned int *)(v30 + 80);
LABEL_39:
                  v45 = *(_QWORD *)(v30 + 40);
                  v44 = *(_QWORD *)(v30 + 48);
                  v34 = *(_QWORD *)(v30 + 32);
                  goto LABEL_30;
                case 6:
                case 7:
                case 8:
                  v32 = *(__int64 **)(v30 + 72);
                  v33 = *(unsigned int *)(v30 + 80);
                  goto LABEL_29;
                case 0xC:
                  v32 = *(__int64 **)(v30 + 96);
                  v33 = *(unsigned int *)(v30 + 104);
LABEL_29:
                  v45 = *(_QWORD *)(v30 + 40);
                  v44 = *(_QWORD *)(v30 + 48);
                  v34 = 0;
                  goto LABEL_30;
                case 0xD:
                  v32 = *(__int64 **)(v30 + 72);
                  v33 = *(unsigned int *)(v30 + 80);
                  v45 = *(_QWORD *)(v30 + 40);
                  v34 = 0;
                  v44 = *(_QWORD *)(v30 + 48);
LABEL_30:
                  v35 = v32;
                  v46 = &v32[3 * v33];
                  if ( v32 != v46 )
                  {
                    do
                    {
                      v47 = 0u;
                      v48 = 0;
                      v49 = 0;
                      sub_E5E2B0(&v50, v31, v30, v35, v34);
                      v36 = *(_QWORD *)(v31 + 8);
                      v37 = _mm_loadu_si128(&v51);
                      v48 = v52;
                      v47 = v37;
                      v49 = v53;
                      v38 = v35;
                      v35 += 3;
                      (*(void (__fastcall **)(__int64, __int64, __int64 *, __m128i *, __int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v36 + 112LL))(
                        v36,
                        v31,
                        v38,
                        &v47,
                        v45,
                        v44,
                        v50.m128i_i64[1],
                        v50.m128i_u8[0],
                        v34);
                    }
                    while ( v46 != v35 );
                  }
                  break;
                default:
                  goto LABEL_32;
              }
              break;
            }
LABEL_32:
            v30 = *(_QWORD *)v30;
          }
          while ( v30 );
LABEL_33:
          v28 = v31;
        }
        result = ++v43;
      }
      while ( v41 != v43 );
    }
  }
  return result;
}
