// Function: sub_1A2A4F0
// Address: 0x1a2a4f0
//
__int64 __fastcall sub_1A2A4F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r12
  char v9; // di
  __int64 v10; // rsi
  int v11; // edx
  unsigned int v12; // r8d
  __int64 *v13; // rax
  __int64 v14; // r11
  __int64 v15; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int8 v19; // al
  __int64 v20; // rax
  char v21; // r8
  unsigned int v22; // edi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // r12
  __int64 *v27; // rax
  int v28; // eax
  __int64 v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // r9
  __int64 v32; // rcx
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  _QWORD *v37; // rdi
  int v38; // ebx
  __int64 v39; // [rsp+10h] [rbp-80h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  __int64 v42; // [rsp+20h] [rbp-70h]
  __int64 v43; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v45[2]; // [rsp+30h] [rbp-60h] BYREF
  __m128i v46; // [rsp+40h] [rbp-50h] BYREF
  __int16 v47; // [rsp+50h] [rbp-40h]

  v8 = a1;
  v44 = a1;
  v9 = *(_BYTE *)(a5 + 8) & 1;
  if ( v9 )
  {
    v10 = a5 + 16;
    v11 = 3;
  }
  else
  {
    v17 = *(unsigned int *)(a5 + 24);
    v10 = *(_QWORD *)(a5 + 16);
    if ( !(_DWORD)v17 )
      goto LABEL_12;
    v11 = v17 - 1;
  }
  v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = (__int64 *)(v10 + 16LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
    goto LABEL_4;
  v28 = 1;
  while ( v14 != -8 )
  {
    v38 = v28 + 1;
    v12 = v11 & (v28 + v12);
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_4;
    v28 = v38;
  }
  if ( v9 )
  {
    v18 = 64;
    goto LABEL_13;
  }
  v17 = *(unsigned int *)(a5 + 24);
LABEL_12:
  v18 = 16 * v17;
LABEL_13:
  v13 = (__int64 *)(v10 + v18);
LABEL_4:
  v15 = 64;
  if ( !v9 )
    v15 = 16LL * *(unsigned int *)(a5 + 24);
  if ( v13 != (__int64 *)(v10 + v15) )
    return v13[1];
  v19 = *(_BYTE *)(v8 + 16);
  if ( v19 <= 0x17u )
  {
    v27 = sub_1A2A380(a5, &v44);
    v8 = v44;
    v27[1] = v44;
  }
  else if ( *(_QWORD *)(v8 + 40) == a3 )
  {
    if ( v19 == 77 )
    {
      v20 = 0x17FFFFFFE8LL;
      v21 = *(_BYTE *)(v8 + 23) & 0x40;
      v22 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
      if ( v22 )
      {
        v23 = 24LL * *(unsigned int *)(v8 + 56) + 8;
        v24 = 0;
        do
        {
          v25 = v8 - 24LL * v22;
          if ( v21 )
            v25 = *(_QWORD *)(v8 - 8);
          if ( a4 == *(_QWORD *)(v25 + v23) )
          {
            v20 = 24 * v24;
            goto LABEL_24;
          }
          ++v24;
          v23 += 8;
        }
        while ( v22 != (_DWORD)v24 );
        v20 = 0x17FFFFFFE8LL;
      }
LABEL_24:
      if ( v21 )
        v26 = *(_QWORD *)(v8 - 8);
      else
        v26 = v8 - 24LL * v22;
      v8 = *(_QWORD *)(v26 + v20);
      sub_1A2A380(a5, &v44)[1] = v8;
    }
    else
    {
      v42 = a3;
      v29 = sub_15F4880(v8);
      v45[0] = sub_1649960(v8);
      v46.m128i_i64[0] = (__int64)v45;
      v46.m128i_i64[1] = (__int64)".st.speculate";
      v45[1] = v30;
      v47 = 773;
      sub_164B780(v29, v46.m128i_i64);
      v31 = v42;
      v32 = a4;
      if ( (*(_DWORD *)(v8 + 20) & 0xFFFFFFF) != 0 )
      {
        v33 = 0;
        v39 = 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
        do
        {
          if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
            v34 = *(_QWORD *)(v8 - 8);
          else
            v34 = v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
          v41 = v32;
          v43 = v31;
          v35 = sub_1A2A4F0(*(_QWORD *)(v34 + v33), a2, v31, v32, a5);
          if ( (*(_BYTE *)(v29 + 23) & 0x40) != 0 )
            v36 = *(_QWORD *)(v29 - 8);
          else
            v36 = v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF);
          v37 = (_QWORD *)(v33 + v36);
          v33 += 24;
          sub_1593B40(v37, v35);
          v31 = v43;
          v32 = v41;
        }
        while ( v33 != v39 );
      }
      v8 = v29;
      v47 = 257;
      sub_1A1C7B0(a2, (_QWORD *)v29, &v46);
      sub_1A2A380(a5, &v44)[1] = v29;
    }
  }
  else
  {
    sub_1A2A380(a5, &v44)[1] = v8;
  }
  return v8;
}
