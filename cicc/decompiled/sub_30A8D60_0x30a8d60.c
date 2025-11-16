// Function: sub_30A8D60
// Address: 0x30a8d60
//
__int64 __fastcall sub_30A8D60(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // r12
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 v6; // rcx
  int v7; // r11d
  _QWORD *v8; // rdi
  unsigned int v9; // edx
  _QWORD *v10; // rax
  __int64 v11; // r9
  _QWORD *v12; // rax
  __int64 v13; // r8
  unsigned int v14; // edx
  int v15; // eax
  _QWORD *v16; // r10
  int v17; // r11d
  unsigned int v18; // edx
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // rsi
  void (__fastcall *v22)(__int64 (__fastcall ***)(__int64), __int64); // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  int v27; // r11d
  __int64 *v28; // [rsp+18h] [rbp-88h] BYREF
  _QWORD v29[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v30; // [rsp+30h] [rbp-70h] BYREF
  __int64 v31; // [rsp+38h] [rbp-68h]
  __int64 v32; // [rsp+40h] [rbp-60h]
  unsigned int v33; // [rsp+48h] [rbp-58h]
  __int64 (__fastcall **v34[2])(__int64); // [rsp+50h] [rbp-50h] BYREF
  __int64 (__fastcall *v35)(const __m128i **, const __m128i *, int); // [rsp+60h] [rbp-40h]
  __int64 (__fastcall *v36)(__int64 (__fastcall ***)(__int64), __int64); // [rsp+68h] [rbp-38h]

  v1 = a1 + 104;
  v3 = *(_QWORD *)(a1 + 120);
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  if ( v3 != a1 + 104 )
  {
    v4 = 0;
    v5 = 0;
    while ( 1 )
    {
      if ( v4 )
      {
        v6 = *(_QWORD *)(v3 + 32);
        v7 = 1;
        v8 = 0;
        v9 = (v4 - 1) & (((0xBF58476D1CE4E5B9LL * v6) >> 31) ^ (484763065 * *(_DWORD *)(v3 + 32)));
        v10 = (_QWORD *)(v5 + 16LL * v9);
        v11 = *v10;
        if ( v6 == *v10 )
        {
LABEL_4:
          v12 = v10 + 1;
          goto LABEL_5;
        }
        while ( v11 != -1 )
        {
          if ( !v8 && v11 == -2 )
            v8 = v10;
          v9 = (v4 - 1) & (v7 + v9);
          v10 = (_QWORD *)(v5 + 16LL * v9);
          v11 = *v10;
          if ( v6 == *v10 )
            goto LABEL_4;
          ++v7;
        }
        if ( !v8 )
          v8 = v10;
        ++v30;
        v15 = v32 + 1;
        if ( 4 * ((int)v32 + 1) < 3 * v4 )
        {
          if ( v4 - (v15 + HIDWORD(v32)) > v4 >> 3 )
            goto LABEL_11;
          sub_30A8B60((__int64)&v30, v4);
          if ( !v33 )
          {
LABEL_51:
            LODWORD(v32) = v32 + 1;
            BUG();
          }
          v13 = *(_QWORD *)(v3 + 32);
          v16 = 0;
          v17 = 1;
          v18 = (v33 - 1) & (((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * *(_DWORD *)(v3 + 32)));
          v15 = v32 + 1;
          v8 = (_QWORD *)(v31 + 16LL * v18);
          v6 = *v8;
          if ( v13 == *v8 )
            goto LABEL_11;
          while ( v6 != -1 )
          {
            if ( !v16 && v6 == -2 )
              v16 = v8;
            v18 = (v33 - 1) & (v17 + v18);
            v8 = (_QWORD *)(v31 + 16LL * v18);
            v6 = *v8;
            if ( v13 == *v8 )
              goto LABEL_11;
            ++v17;
          }
          goto LABEL_27;
        }
      }
      else
      {
        ++v30;
      }
      sub_30A8B60((__int64)&v30, 2 * v4);
      if ( !v33 )
        goto LABEL_51;
      v13 = *(_QWORD *)(v3 + 32);
      v14 = (v33 - 1) & (((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * *(_DWORD *)(v3 + 32)));
      v15 = v32 + 1;
      v8 = (_QWORD *)(v31 + 16LL * v14);
      v6 = *v8;
      if ( v13 == *v8 )
        goto LABEL_11;
      v27 = 1;
      v16 = 0;
      while ( v6 != -1 )
      {
        if ( !v16 && v6 == -2 )
          v16 = v8;
        v14 = (v33 - 1) & (v27 + v14);
        v8 = (_QWORD *)(v31 + 16LL * v14);
        v6 = *v8;
        if ( v13 == *v8 )
          goto LABEL_11;
        ++v27;
      }
LABEL_27:
      v6 = v13;
      if ( v16 )
        v8 = v16;
LABEL_11:
      LODWORD(v32) = v15;
      if ( *v8 != -1 )
        --HIDWORD(v32);
      *v8 = v6;
      v12 = v8 + 1;
      v8[1] = 0;
LABEL_5:
      *v12 = v3 + 80;
      v3 = sub_220EEE0(v3);
      if ( v1 == v3 )
        break;
      v5 = v31;
      v4 = v33;
    }
  }
  v19 = *(_QWORD *)(a1 + 24);
  v20 = a1 + 8;
  v34[1] = (__int64 (__fastcall **)(__int64))v34;
  v28 = &v30;
  v21 = v19 + 40;
  v29[0] = sub_30A68F0;
  v29[1] = &v28;
  v34[0] = (__int64 (__fastcall **)(__int64))v29;
  v22 = (void (__fastcall *)(__int64 (__fastcall ***)(__int64), __int64))sub_30A6E70;
  v36 = sub_30A6E70;
  v35 = sub_30A6890;
  if ( v19 == v20 )
  {
    sub_30A6890((const __m128i **)v34, (const __m128i *)v34, 3);
  }
  else
  {
    while ( 1 )
    {
      v22(v34, v21);
      v23 = v19;
      v24 = sub_220EEE0(v19);
      v19 = v24;
      if ( v20 == v24 )
        break;
      v21 = v24 + 40;
      if ( !v35 )
        sub_4263D6(v23, v21, v25);
      v22 = (void (__fastcall *)(__int64 (__fastcall ***)(__int64), __int64))v36;
    }
    if ( v35 )
      v35((const __m128i **)v34, (const __m128i *)v34, 3);
  }
  return sub_C7D6A0(v31, 16LL * v33, 8);
}
