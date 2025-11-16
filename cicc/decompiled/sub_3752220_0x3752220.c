// Function: sub_3752220
// Address: 0x3752220
//
__int64 __fastcall sub_3752220(
        __int64 *a1,
        __int64 *a2,
        unsigned __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8,
        unsigned __int8 a9,
        unsigned __int8 a10)
{
  unsigned int v10; // r15d
  bool v13; // r12
  unsigned __int16 *v15; // r11
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdi
  int v20; // r12d
  bool v21; // dl
  char v22; // cl
  __int64 v23; // rsi
  __int64 v25; // rax
  _QWORD *v26; // r15
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // r8
  __int64 v30; // r9
  __int32 v31; // eax
  __int64 *v32; // r10
  unsigned __int16 *v33; // r11
  __int64 v34; // rcx
  __int64 *v35; // rsi
  __int64 v36; // rdi
  _QWORD *v37; // rax
  __int64 v38; // rdx
  int v39; // edx
  __int64 v40; // rax
  bool v41; // zf
  __int64 i; // rdx
  _BYTE *v43; // rcx
  __int64 v44; // [rsp+8h] [rbp-C8h]
  unsigned __int16 *v45; // [rsp+10h] [rbp-C0h]
  unsigned __int16 *v47; // [rsp+28h] [rbp-A8h]
  unsigned __int16 *v48; // [rsp+30h] [rbp-A0h]
  __int32 v49; // [rsp+30h] [rbp-A0h]
  int v50; // [rsp+38h] [rbp-98h]
  unsigned __int16 *v51; // [rsp+38h] [rbp-98h]
  unsigned __int8 *v52; // [rsp+48h] [rbp-88h] BYREF
  __int64 v53[4]; // [rsp+50h] [rbp-80h] BYREF
  __m128i v54; // [rsp+70h] [rbp-60h] BYREF
  __int64 v55; // [rsp+80h] [rbp-50h]
  __int64 v56; // [rsp+88h] [rbp-48h]
  __int64 v57; // [rsp+90h] [rbp-40h]

  v10 = a5;
  v13 = 0;
  v50 = sub_3752000(a1, a3, a4, a7, a5, a6);
  v15 = *(unsigned __int16 **)(a2[1] + 16);
  if ( v10 < v15[1] )
    v13 = (v15[20 * *v15 + 21 + 3 * v10 + 3 * (unsigned __int64)v15[8]] & 4) != 0;
  if ( a6 )
  {
    if ( v10 < *(unsigned __int16 *)(a6 + 2) )
    {
      v48 = *(unsigned __int16 **)(a2[1] + 16);
      v25 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64))(*(_QWORD *)a1[2] + 16LL))(
              a1[2],
              a6,
              v10,
              a1[3],
              *a1);
      v15 = v48;
      v26 = (_QWORD *)v25;
      if ( v25 )
      {
        v45 = v48;
        v27 = sub_2EBE590(a1[1], v50, v25, 4 * (unsigned int)(*(_DWORD *)(a3 + 24) != -11));
        v15 = v48;
        if ( !v27 )
        {
          v28 = sub_2FF6410(a1[3], v26);
          v31 = sub_2EC06C0(a1[1], (__int64)v28, byte_3F871B3, 0, v29, v30);
          v32 = a1;
          v33 = v48;
          v49 = v31;
          v34 = *(_QWORD *)(a1[2] + 8) - 800LL;
          v52 = *(unsigned __int8 **)(a2[1] + 56);
          if ( v52 )
          {
            v44 = v34;
            sub_B96E90((__int64)&v52, (__int64)v52, 1);
            v33 = v45;
            v34 = v44;
            v53[0] = (__int64)v52;
            v32 = a1;
            if ( v52 )
            {
              sub_B976B0((__int64)&v52, v52, (__int64)v53);
              v32 = a1;
              v52 = 0;
              v34 = v44;
              v33 = v45;
            }
          }
          else
          {
            v53[0] = 0;
          }
          v35 = (__int64 *)v32[6];
          v36 = v32[5];
          v47 = v33;
          v53[1] = 0;
          v53[2] = 0;
          v37 = sub_2F26260(v36, v35, v53, v34, v49);
          v54.m128i_i64[0] = 0;
          v55 = 0;
          v54.m128i_i32[2] = v50;
          v56 = 0;
          v57 = 0;
          sub_2E8EAD0(v38, (__int64)v37, &v54);
          v15 = v47;
          if ( v53[0] )
          {
            sub_B91220((__int64)v53, v53[0]);
            v15 = v47;
          }
          if ( v52 )
          {
            v51 = v15;
            sub_B91220((__int64)&v52, (__int64)v52);
            v15 = v51;
          }
          v50 = v49;
        }
      }
    }
  }
  v16 = *(_QWORD *)(a3 + 56);
  if ( !v16 )
  {
LABEL_14:
    v19 = a2[1];
    goto LABEL_15;
  }
  v17 = 1;
  do
  {
    while ( a4 != *(_DWORD *)(v16 + 8) )
    {
      v16 = *(_QWORD *)(v16 + 32);
      if ( !v16 )
        goto LABEL_13;
    }
    if ( !v17 )
      goto LABEL_14;
    v18 = *(_QWORD *)(v16 + 32);
    if ( !v18 )
      goto LABEL_28;
    if ( a4 == *(_DWORD *)(v18 + 8) )
      goto LABEL_14;
    v16 = *(_QWORD *)(v18 + 32);
    v17 = 0;
  }
  while ( v16 );
LABEL_13:
  if ( v17 == 1 )
    goto LABEL_14;
LABEL_28:
  v39 = *(_DWORD *)(a3 + 24);
  if ( v39 >= 0 )
  {
    if ( (unsigned int)(v39 - 493) <= 3 )
      BUG();
    if ( v39 != 50 )
    {
      v40 = a2[1];
LABEL_36:
      v19 = v40;
      if ( !a8 )
      {
        if ( !(a9 | a10) )
        {
          for ( i = *(_DWORD *)(v40 + 40) & 0xFFFFFF; (_DWORD)i; i = (unsigned int)(i - 1) )
          {
            v43 = (_BYTE *)(*(_QWORD *)(v40 + 32) + 40LL * (unsigned int)(i - 1));
            if ( *v43 || (v43[3] & 0x20) == 0 )
              break;
          }
          if ( v15[1] <= (unsigned int)i || (v15[20 * *v15 + 22 + 3 * v15[8] + 3 * i] & 1) == 0 )
          {
            v21 = v13;
            v22 = 1;
            v13 = 0;
            goto LABEL_17;
          }
        }
        if ( !v13 )
        {
          v22 = 0;
          v21 = 0;
          goto LABEL_17;
        }
        v20 = 2;
        goto LABEL_16;
      }
      if ( v13 )
      {
        v21 = v13;
        v22 = 0;
        goto LABEL_17;
      }
      v20 = 0;
LABEL_34:
      v41 = v20 == 0;
      v13 = 1;
      v21 = !v41;
      v22 = 0;
      goto LABEL_17;
    }
    goto LABEL_14;
  }
  v19 = a2[1];
  v40 = v19;
  if ( (unsigned int)(-v39 - 47) > 3 )
    goto LABEL_36;
LABEL_15:
  v20 = 2 * v13;
  if ( a8 )
    goto LABEL_34;
LABEL_16:
  v21 = v20 != 0;
  v22 = 0;
  v13 = 0;
LABEL_17:
  v23 = *a2;
  v54.m128i_i8[0] = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v54.m128i_i8[3] = (v22 << 6) & 0x7F | (16 * v21) & 0x7F | v54.m128i_i8[3] & 0xF;
  v54.m128i_i16[1] &= 0xF00Fu;
  v54.m128i_i32[0] &= 0xFFF000FF;
  v54.m128i_i8[4] = (8 * v13) | v54.m128i_i8[4] & 0xF0;
  v54.m128i_i32[2] = v50;
  return sub_2E8EAD0(v19, v23, &v54);
}
