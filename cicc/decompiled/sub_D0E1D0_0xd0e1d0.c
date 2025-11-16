// Function: sub_D0E1D0
// Address: 0xd0e1d0
//
__int64 __fastcall sub_D0E1D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  int v10; // edx
  unsigned __int64 v11; // rax
  _QWORD *v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rbx
  unsigned __int64 v15; // rax
  int v16; // eax
  unsigned int v17; // esi
  __int64 v18; // rdi
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r15
  __int64 *v24; // rax
  char v25; // al
  __int64 *v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // edx
  __int64 *v32; // rcx
  __int64 v33; // rcx
  __int64 *v34; // rax
  unsigned __int64 v35; // rax
  int v36; // edx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // r8
  char *v40; // rcx
  const __m128i *v41; // rdx
  __m128i *v42; // rax
  char *v43; // rbx
  _QWORD v44[2]; // [rsp+10h] [rbp-1E0h] BYREF
  int v45; // [rsp+20h] [rbp-1D0h]
  __int64 v46; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 *v47; // [rsp+38h] [rbp-1B8h]
  __int64 v48; // [rsp+40h] [rbp-1B0h]
  int v49; // [rsp+48h] [rbp-1A8h]
  char v50; // [rsp+4Ch] [rbp-1A4h]
  __int64 v51; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v52; // [rsp+90h] [rbp-160h] BYREF
  __int64 *v53; // [rsp+98h] [rbp-158h]
  __int64 v54; // [rsp+A0h] [rbp-150h]
  int v55; // [rsp+A8h] [rbp-148h]
  char v56; // [rsp+ACh] [rbp-144h]
  __int64 v57; // [rsp+B0h] [rbp-140h] BYREF
  char *v58; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v59; // [rsp+F8h] [rbp-F8h]
  _QWORD v60[2]; // [rsp+100h] [rbp-F0h] BYREF
  int v61; // [rsp+110h] [rbp-E0h]

  v2 = *(_QWORD *)(a1 + 80);
  if ( !v2 )
    BUG();
  v3 = v2 + 24;
  result = *(_QWORD *)(v2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 + 24 != result )
  {
    if ( !result )
LABEL_43:
      BUG();
    v5 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result <= 0xA )
    {
      result = sub_B46E30(v5);
      if ( (_DWORD)result )
      {
        v7 = v2 - 24;
        v55 = 0;
        v47 = &v51;
        v58 = (char *)v60;
        v59 = 0x800000000LL;
        v53 = &v57;
        v56 = 1;
        v48 = 0x100000008LL;
        v49 = 0;
        v50 = 1;
        v51 = v2 - 24;
        v8 = *(_QWORD *)(v2 + 24);
        v52 = 0;
        v54 = 8;
        v9 = v8 & 0xFFFFFFFFFFFFFFF8LL;
        v46 = 1;
        if ( v3 == v9 )
        {
          v11 = 0;
        }
        else
        {
          if ( !v9 )
LABEL_76:
            BUG();
          v10 = *(unsigned __int8 *)(v9 - 24);
          v11 = v9 - 24;
          if ( (unsigned int)(v10 - 30) >= 0xB )
            v11 = 0;
        }
        v60[0] = v2 - 24;
        v12 = v60;
        v61 = 0;
        LODWORD(v59) = 1;
        HIDWORD(v54) = 1;
        v57 = v2 - 24;
        v52 = 1;
        v60[1] = v11;
        v13 = 1;
        while ( 1 )
        {
          v14 = (__int64)&v12[3 * v13 - 3];
LABEL_12:
          v15 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v15 == v3 )
            goto LABEL_30;
LABEL_13:
          if ( !v15 )
            goto LABEL_43;
          if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 <= 0xA )
          {
            v16 = sub_B46E30(v15 - 24);
            v17 = *(_DWORD *)(v14 + 16);
            if ( v17 == v16 )
              goto LABEL_31;
            goto LABEL_16;
          }
LABEL_30:
          while ( 1 )
          {
            v17 = *(_DWORD *)(v14 + 16);
            if ( !v17 )
              break;
LABEL_16:
            v18 = *(_QWORD *)(v14 + 8);
            *(_DWORD *)(v14 + 16) = v17 + 1;
            v23 = sub_B46EC0(v18, v17);
            if ( v50 )
            {
              v24 = v47;
              v20 = HIDWORD(v48);
              v19 = &v47[HIDWORD(v48)];
              if ( v47 != v19 )
              {
                while ( v23 != *v24 )
                {
                  if ( v19 == ++v24 )
                    goto LABEL_58;
                }
                v25 = v56;
                goto LABEL_22;
              }
LABEL_58:
              if ( HIDWORD(v48) < (unsigned int)v48 )
              {
                v33 = (unsigned int)++HIDWORD(v48);
                *v19 = v23;
                ++v46;
                if ( !v56 )
                  goto LABEL_60;
LABEL_46:
                v34 = v53;
                v33 = HIDWORD(v54);
                v19 = &v53[HIDWORD(v54)];
                if ( v53 == v19 )
                {
LABEL_63:
                  if ( HIDWORD(v54) >= (unsigned int)v54 )
                  {
LABEL_60:
                    sub_C8CC70((__int64)&v52, v23, (__int64)v19, v33, v21, v22);
                  }
                  else
                  {
                    ++HIDWORD(v54);
                    *v19 = v23;
                    ++v52;
                  }
                }
                else
                {
                  while ( v23 != *v34 )
                  {
                    if ( v19 == ++v34 )
                      goto LABEL_63;
                  }
                }
                v35 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v35 == v23 + 48 )
                {
                  v37 = 0;
                }
                else
                {
                  if ( !v35 )
                    goto LABEL_76;
                  v36 = *(unsigned __int8 *)(v35 - 24);
                  v37 = v35 - 24;
                  if ( (unsigned int)(v36 - 30) >= 0xB )
                    v37 = 0;
                }
                v44[1] = v37;
                v38 = (unsigned int)v59;
                v30 = HIDWORD(v59);
                v44[0] = v23;
                v39 = (unsigned int)v59 + 1LL;
                v40 = v58;
                v45 = 0;
                v41 = (const __m128i *)v44;
                if ( v39 > HIDWORD(v59) )
                {
                  if ( v58 > (char *)v44 || v44 >= (_QWORD *)&v58[24 * (unsigned int)v59] )
                  {
                    v30 = (__int64)v60;
                    sub_C8D5F0((__int64)&v58, v60, (unsigned int)v59 + 1LL, 0x18u, v39, v22);
                    v40 = v58;
                    v38 = (unsigned int)v59;
                    v41 = (const __m128i *)v44;
                  }
                  else
                  {
                    v30 = (__int64)v60;
                    v43 = (char *)((char *)v44 - v58);
                    sub_C8D5F0((__int64)&v58, v60, (unsigned int)v59 + 1LL, 0x18u, v39, v22);
                    v40 = v58;
                    v38 = (unsigned int)v59;
                    v41 = (const __m128i *)&v43[(_QWORD)v58];
                  }
                }
                v42 = (__m128i *)&v40[24 * v38];
                *v42 = _mm_loadu_si128(v41);
                v42[1].m128i_i64[0] = v41[1].m128i_i64[0];
                result = (unsigned int)v59;
                v31 = v59 + 1;
                LODWORD(v59) = v59 + 1;
                goto LABEL_56;
              }
            }
            sub_C8CC70((__int64)&v46, v23, (__int64)v19, v20, v21, v22);
            v25 = v56;
            if ( (_BYTE)v19 )
            {
              if ( !v56 )
                goto LABEL_60;
              goto LABEL_46;
            }
LABEL_22:
            if ( v25 )
            {
              v26 = v53;
              v27 = &v53[HIDWORD(v54)];
              if ( v53 == v27 )
                goto LABEL_12;
              while ( v23 != *v26 )
              {
                if ( v27 == ++v26 )
                  goto LABEL_12;
              }
            }
            else if ( !sub_C8CA60((__int64)&v52, v23) )
            {
              goto LABEL_12;
            }
            v28 = *(unsigned int *)(a2 + 8);
            if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
            {
              sub_C8D5F0(a2, (const void *)(a2 + 16), v28 + 1, 0x10u, v21, v22);
              v28 = *(unsigned int *)(a2 + 8);
            }
            v29 = (_QWORD *)(*(_QWORD *)a2 + 16 * v28);
            *v29 = v7;
            v29[1] = v23;
            ++*(_DWORD *)(a2 + 8);
            v15 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v15 != v3 )
              goto LABEL_13;
          }
LABEL_31:
          v30 = *(_QWORD *)&v58[24 * (unsigned int)v59 - 24];
          v31 = v59 - 1;
          LODWORD(v59) = v59 - 1;
          if ( !v56 )
          {
            result = (__int64)sub_C8CA60((__int64)&v52, v30);
            if ( result )
            {
              *(_QWORD *)result = -2;
              ++v55;
              ++v52;
            }
            v31 = v59;
            goto LABEL_56;
          }
          result = HIDWORD(v54);
          v32 = &v53[HIDWORD(v54)];
          if ( v53 == v32 )
          {
LABEL_56:
            if ( !v31 )
              goto LABEL_37;
          }
          else
          {
            result = (__int64)v53;
            while ( v30 != *(_QWORD *)result )
            {
              result += 8;
              if ( v32 == (__int64 *)result )
                goto LABEL_56;
            }
            --HIDWORD(v54);
            *(_QWORD *)result = v53[HIDWORD(v54)];
            v31 = v59;
            ++v52;
            if ( !(_DWORD)v59 )
            {
LABEL_37:
              if ( !v56 )
                result = _libc_free(v53, v30);
              if ( v58 != (char *)v60 )
                result = _libc_free(v58, v30);
              if ( !v50 )
                return _libc_free(v47, v30);
              return result;
            }
          }
          v12 = v58;
          v13 = v31;
          v7 = *(_QWORD *)&v58[24 * v31 - 24];
          v3 = v7 + 48;
        }
      }
    }
  }
  return result;
}
