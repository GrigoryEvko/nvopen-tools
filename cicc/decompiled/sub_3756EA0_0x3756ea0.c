// Function: sub_3756EA0
// Address: 0x3756ea0
//
__int64 __fastcall sub_3756EA0(
        _QWORD *a1,
        unsigned __int64 a2,
        __int64 *a3,
        unsigned __int16 *a4,
        char a5,
        char a6,
        __m128i *a7)
{
  __int64 (*v10)(); // rax
  __int64 result; // rax
  __int64 i; // rbx
  _QWORD *v13; // rax
  _QWORD *v14; // r8
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 v17; // r10
  unsigned int v18; // r11d
  __int64 (__fastcall *v19)(__int64, unsigned __int16); // rcx
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int32 v25; // eax
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int32 v29; // eax
  __int64 v30; // rsi
  __m128i *v31; // r8
  int v32; // ecx
  int v33; // r10d
  unsigned int k; // edx
  __int8 *v35; // rsi
  unsigned int v36; // edx
  __int64 j; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // rsi
  __int32 v43; // ecx
  unsigned __int32 v44; // edx
  __int64 v45; // rax
  unsigned __int8 v46; // al
  __int64 (__fastcall *v47)(__int64, unsigned __int16); // [rsp+10h] [rbp-C0h]
  int v48; // [rsp+1Ch] [rbp-B4h]
  _QWORD *v49; // [rsp+20h] [rbp-B0h]
  __int64 v50; // [rsp+20h] [rbp-B0h]
  __int64 v52; // [rsp+38h] [rbp-98h]
  __int64 v53; // [rsp+40h] [rbp-90h]
  _QWORD *v54; // [rsp+40h] [rbp-90h]
  unsigned __int32 v55; // [rsp+40h] [rbp-90h]
  unsigned __int32 v56; // [rsp+40h] [rbp-90h]
  int v57; // [rsp+40h] [rbp-90h]
  _QWORD *v58; // [rsp+40h] [rbp-90h]
  _QWORD *v59; // [rsp+40h] [rbp-90h]
  unsigned int v62; // [rsp+4Ch] [rbp-84h]
  unsigned __int64 v63; // [rsp+50h] [rbp-80h] BYREF
  int v64; // [rsp+58h] [rbp-78h]
  int v65[4]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v66; // [rsp+70h] [rbp-60h] BYREF
  __int64 v67; // [rsp+80h] [rbp-50h]
  __int64 v68; // [rsp+88h] [rbp-48h]
  __int64 v69; // [rsp+90h] [rbp-40h]

  v62 = sub_3751FC0(a2);
  v10 = *(__int64 (**)())(**(_QWORD **)(*a1 + 8LL) + 224LL);
  if ( v10 == sub_23CE3C0 || (unsigned __int8)v10() || (*((_QWORD *)a4 + 3) & 0x8000000002LL) != 0x8000000002LL )
    result = *((unsigned __int8 *)a4 + 4);
  else
    result = v62;
  if ( *(_DWORD *)(a2 + 24) == -33 )
    result = v62;
  if ( !(_DWORD)result )
    return result;
  v52 = (unsigned int)result;
  v48 = (a2 >> 9) ^ (a2 >> 4);
  for ( i = 0; i != v52; ++i )
  {
    v53 = a1[3];
    v13 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD, unsigned __int16 *, _QWORD, __int64, _QWORD))(*(_QWORD *)a1[2] + 16LL))(
                      a1[2],
                      a4,
                      (unsigned int)i,
                      v53,
                      *a1);
    v14 = sub_2FF6410(v53, v13);
    if ( v62 > (unsigned int)i )
    {
      v16 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * i);
      if ( (_WORD)v16 )
      {
        v17 = a1[4];
        if ( *(_QWORD *)(v17 + 8LL * (unsigned __int16)v16 + 112) )
        {
          v18 = 1;
          v19 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v17 + 552LL);
          if ( (*(_BYTE *)(a2 + 32) & 4) == 0 )
          {
            v18 = 0;
            if ( v14 )
            {
              v20 = a1[3];
              v21 = *(__int64 (**)())(*(_QWORD *)v20 + 176LL);
              if ( v21 != sub_2FF51F0 )
              {
                v47 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v17 + 552LL);
                v50 = a1[4];
                v59 = v14;
                v46 = ((__int64 (__fastcall *)(__int64, _QWORD *))v21)(v20, v14);
                v19 = v47;
                v18 = v46;
                v17 = v50;
                v14 = v59;
                v16 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * i);
              }
            }
          }
          if ( v19 == sub_2EC09E0 )
          {
            v22 = *(_QWORD *)(v17 + 8 * v16 + 112);
          }
          else
          {
            v58 = v14;
            v45 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v19)(v17, v16, v18);
            v14 = v58;
            v22 = v45;
          }
          if ( v14 )
          {
            v54 = v14;
            v23 = sub_2FF6970(a1[3], (__int64)v14, v22);
            v14 = v54;
            v22 = v23;
          }
          if ( v22 )
            v14 = (_QWORD *)v22;
        }
      }
    }
    if ( !a4[1] )
      goto LABEL_22;
    if ( (a4[20 * *a4 + 21 + 3 * a4[8] + 3 * i] & 4) == 0 )
      goto LABEL_22;
    v49 = v14;
    v28 = a3[1];
    v29 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL * ((unsigned int)i - v62)) + 96LL);
    v66.m128i_i64[0] = 0x10000000;
    v30 = *a3;
    v66.m128i_i32[2] = v29;
    v56 = v29;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    sub_2E8EAD0(v28, v30, &v66);
    result = v56;
    v14 = v49;
    if ( !v56 )
    {
LABEL_22:
      v24 = a1[1];
      if ( !a5 && !a6 )
      {
        for ( j = *(_QWORD *)(a2 + 56); j; j = *(_QWORD *)(j + 32) )
        {
          v38 = *(_QWORD *)(j + 16);
          if ( *(_DWORD *)(v38 + 24) == 49 )
          {
            v39 = *(_QWORD *)(v38 + 40);
            if ( a2 == *(_QWORD *)(v39 + 80) && *(_DWORD *)(v39 + 88) == (_DWORD)i )
            {
              v40 = *(_DWORD *)(*(_QWORD *)(v39 + 40) + 96LL);
              if ( v40 < 0
                && v14 == (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v24 + 56) + 16LL * (v40 & 0x7FFFFFFF))
                                   & 0xFFFFFFFFFFFFFFF8LL) )
              {
                v66.m128i_i64[0] = 0x10000000;
                v67 = 0;
                v41 = a3[1];
                v42 = *a3;
                v66.m128i_i32[2] = v40;
                v57 = v40;
                v68 = 0;
                v69 = 0;
                result = sub_2E8EAD0(v41, v42, &v66);
                if ( v62 > (unsigned int)i )
                {
                  LODWORD(result) = v57;
                  goto LABEL_27;
                }
                goto LABEL_28;
              }
            }
          }
        }
      }
      v25 = sub_2EC06C0(v24, (__int64)v14, byte_3F871B3, 0, (__int64)v14, v15);
      v66.m128i_i64[0] = 0x10000000;
      v66.m128i_i32[2] = v25;
      v26 = a3[1];
      v27 = *a3;
      v55 = v25;
      v67 = 0;
      v68 = 0;
      v69 = 0;
      sub_2E8EAD0(v26, v27, &v66);
      result = v55;
    }
    if ( v62 <= (unsigned int)i )
      continue;
    if ( !a5 )
      goto LABEL_27;
    if ( (a7->m128i_i8[8] & 1) != 0 )
    {
      v31 = a7 + 1;
      v32 = 15;
    }
    else
    {
      v43 = a7[1].m128i_i32[2];
      v31 = (__m128i *)a7[1].m128i_i64[0];
      if ( !v43 )
        goto LABEL_27;
      v32 = v43 - 1;
    }
    v33 = 1;
    for ( k = v32 & (i + v48); ; k = v32 & v36 )
    {
      v35 = &v31->m128i_i8[24 * k];
      if ( a2 == *(_QWORD *)v35 )
        break;
      if ( !*(_QWORD *)v35 && *((_DWORD *)v35 + 2) == -1 )
        goto LABEL_27;
LABEL_37:
      v36 = v33 + k;
      ++v33;
    }
    if ( (_DWORD)i != *((_DWORD *)v35 + 2) )
      goto LABEL_37;
    *(_QWORD *)v35 = 0;
    *((_DWORD *)v35 + 2) = -2;
    v44 = a7->m128i_u32[2];
    ++a7->m128i_i32[3];
    a7->m128i_i32[2] = (2 * (v44 >> 1) - 2) | v44 & 1;
LABEL_27:
    v63 = a2;
    v64 = i;
    v65[0] = result;
    result = sub_3755010((__int64)&v66, a7, &v63, v65);
LABEL_28:
    ;
  }
  return result;
}
