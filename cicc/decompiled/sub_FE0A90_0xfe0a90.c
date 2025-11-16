// Function: sub_FE0A90
// Address: 0xfe0a90
//
__int64 __fastcall sub_FE0A90(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  bool v7; // r13
  unsigned int v8; // esi
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  int v11; // ecx
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  const __m128i *v18; // rsi
  __int64 v20; // r9
  unsigned int v21; // edx
  __int64 v22; // rcx
  __int64 v23; // rdi
  int v24; // r11d
  int v25; // edi
  __int64 v26; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v27; // [rsp+8h] [rbp-A8h]
  __int64 v28; // [rsp+8h] [rbp-A8h]
  int v29; // [rsp+1Ch] [rbp-94h] BYREF
  void *v30; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v31[2]; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int64 v32; // [rsp+38h] [rbp-78h]
  __int64 v33; // [rsp+40h] [rbp-70h]
  __int64 v34; // [rsp+50h] [rbp-60h] BYREF
  void *v35; // [rsp+58h] [rbp-58h]
  unsigned __int64 v36; // [rsp+60h] [rbp-50h] BYREF
  __int64 v37; // [rsp+68h] [rbp-48h]
  unsigned __int64 v38; // [rsp+70h] [rbp-40h]
  __int64 v39; // [rsp+78h] [rbp-38h]

  v3 = a1 + 160;
  v36 = a2;
  v34 = 0;
  v35 = 0;
  v7 = a2 != -8192 && a2 != -4096 && a2 != 0;
  if ( v7 )
  {
    sub_BD73F0((__int64)&v34);
    v3 = a1 + 160;
  }
  v8 = *(_DWORD *)(a1 + 184);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 160);
    v30 = 0;
    goto LABEL_5;
  }
  v9 = v36;
  v20 = *(_QWORD *)(a1 + 168);
  v21 = (v8 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
  v22 = v20 + 72LL * v21;
  v23 = *(_QWORD *)(v22 + 16);
  if ( v23 != v36 )
  {
    v24 = 1;
    v10 = 0;
    while ( v23 != -4096 )
    {
      if ( !v10 && v23 == -8192 )
        v10 = v22;
      v21 = (v8 - 1) & (v24 + v21);
      v22 = v20 + 72LL * v21;
      v23 = *(_QWORD *)(v22 + 16);
      if ( v36 == v23 )
        goto LABEL_39;
      ++v24;
    }
    v25 = *(_DWORD *)(a1 + 176);
    if ( !v10 )
      v10 = v22;
    ++*(_QWORD *)(a1 + 160);
    v11 = v25 + 1;
    v30 = (void *)v10;
    if ( 4 * (v25 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 180) - v11 > v8 >> 3 )
      {
LABEL_7:
        *(_DWORD *)(a1 + 176) = v11;
        if ( *(_QWORD *)(v10 + 16) == -4096 )
        {
          if ( v9 == -4096 )
          {
            v12 = -4096;
            goto LABEL_16;
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 180);
          v12 = *(_QWORD *)(v10 + 16);
          if ( v9 == v12 )
            goto LABEL_16;
          if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          {
            v27 = v9;
            sub_BD60C0((_QWORD *)v10);
            v9 = v27;
          }
        }
        *(_QWORD *)(v10 + 16) = v9;
        if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
          sub_BD73F0(v10);
        v12 = v36;
LABEL_16:
        *(_DWORD *)(v10 + 24) = -1;
        *(_OWORD *)(v10 + 48) = 0;
        *(_QWORD *)(v10 + 64) = 0;
        *(_QWORD *)(v10 + 32) = &unk_49E5548;
        *(_QWORD *)(v10 + 40) = 2;
        if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
          sub_BD60C0(&v34);
        v13 = *(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8);
        v32 = a2;
        v31[1] = 0;
        v31[0] = 2;
        v14 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
        v29 = v14;
        if ( v7 )
        {
          sub_BD73F0((__int64)v31);
          v33 = a1;
          v37 = 0;
          v30 = &unk_49E5548;
          v38 = v32;
          LODWORD(v34) = v29;
          v36 = v31[0] & 6;
          if ( v32 != -4096 && v32 != -8192 && v32 )
          {
            sub_BD6050(&v36, v31[0] & 0xFFFFFFFFFFFFFFF8LL);
            v15 = v33;
          }
          else
          {
            v15 = a1;
          }
        }
        else
        {
          v33 = a1;
          v30 = &unk_49E5548;
          v36 = 2;
          v37 = 0;
          v38 = a2;
          LODWORD(v34) = v14;
          v15 = a1;
        }
        v39 = v15;
        v35 = &unk_49E5548;
        v16 = *(_QWORD *)(v10 + 56);
        *(_DWORD *)(v10 + 24) = v34;
        v17 = v38;
        if ( v16 != v38 )
        {
          if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
          {
            sub_BD60C0((_QWORD *)(v10 + 40));
            v17 = v38;
          }
          *(_QWORD *)(v10 + 56) = v17;
          if ( v17 == 0 || v17 == -4096 || v17 == -8192 )
          {
            *(_QWORD *)(v10 + 64) = v39;
LABEL_30:
            v30 = &unk_49DB368;
            if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
              sub_BD60C0(v31);
            v18 = *(const __m128i **)(a1 + 16);
            if ( v18 == *(const __m128i **)(a1 + 24) )
            {
              sub_FDDD10((const __m128i **)(a1 + 8), v18);
            }
            else
            {
              if ( v18 )
              {
                v18->m128i_i64[0] = 0;
                v18->m128i_i16[4] = 0;
                v18[1].m128i_i64[0] = 0;
                v18 = *(const __m128i **)(a1 + 16);
              }
              *(_QWORD *)(a1 + 16) = (char *)v18 + 24;
            }
            return sub_FE8AF0(a1, &v29, a3);
          }
          sub_BD6050((unsigned __int64 *)(v10 + 40), v36 & 0xFFFFFFFFFFFFFFF8LL);
          v16 = v38;
        }
        *(_QWORD *)(v10 + 64) = v39;
        v35 = &unk_49DB368;
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
          sub_BD60C0(&v36);
        goto LABEL_30;
      }
LABEL_6:
      v26 = v3;
      sub_FE0650(v3, v8);
      sub_FDDC50(v26, (__int64)&v34, &v30);
      v9 = v36;
      v10 = (__int64)v30;
      v11 = *(_DWORD *)(a1 + 176) + 1;
      goto LABEL_7;
    }
LABEL_5:
    v8 *= 2;
    goto LABEL_6;
  }
LABEL_39:
  if ( v36 != -4096 && v36 != 0 && v36 != -8192 )
  {
    v28 = v22;
    sub_BD60C0(&v34);
    v22 = v28;
  }
  return sub_FE8AF0(a1, v22 + 24, a3);
}
