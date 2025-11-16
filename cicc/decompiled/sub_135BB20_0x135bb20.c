// Function: sub_135BB20
// Address: 0x135bb20
//
void __fastcall sub_135BB20(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // r14d
  int v7; // edx
  char v8; // r14
  unsigned int v9; // r12d
  unsigned int v10; // edi
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 *v13; // r11
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // r12d
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // r11
  int v24; // r8d
  __int64 v25; // rdi
  int v26; // r10d
  int v27; // [rsp+Ch] [rbp-B4h]
  __int64 v28; // [rsp+10h] [rbp-B0h]
  __int64 v29; // [rsp+28h] [rbp-98h]
  __int64 v30; // [rsp+28h] [rbp-98h]
  __int64 v31; // [rsp+28h] [rbp-98h]
  __int64 *v32; // [rsp+28h] [rbp-98h]
  __int64 *v33; // [rsp+28h] [rbp-98h]
  __int64 *v34; // [rsp+28h] [rbp-98h]
  void *v35; // [rsp+30h] [rbp-90h] BYREF
  _BYTE v36[16]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v37; // [rsp+48h] [rbp-78h]
  __m128i v38; // [rsp+60h] [rbp-60h] BYREF
  __int64 v39; // [rsp+70h] [rbp-50h]
  __int64 v40; // [rsp+78h] [rbp-48h]

  v3 = *(_DWORD *)(a1 + 48);
  if ( v3 )
  {
    v29 = *(_QWORD *)(a1 + 32);
    sub_1359800(&v35, -8, 0);
    sub_1359800(&v38, -16, 0);
    v7 = v3 - 1;
    v8 = 1;
    v9 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v10 = v7 & v9;
    v11 = v29 + 48LL * (v7 & v9);
    v12 = *(_QWORD *)(v11 + 24);
    if ( v12 != a2 )
    {
      v23 = v29 + 48LL * (v7 & v9);
      v24 = 1;
      v11 = 0;
      while ( v37 != v12 )
      {
        if ( v40 != v12 || v11 )
          v23 = v11;
        v10 = v7 & (v24 + v10);
        v11 = v29 + 48LL * v10;
        v12 = *(_QWORD *)(v11 + 24);
        if ( v12 == a2 )
        {
          v8 = 1;
          goto LABEL_4;
        }
        ++v24;
        v11 = v23;
        v23 = v29 + 48LL * v10;
      }
      v8 = 0;
      if ( !v11 )
        v11 = v23;
    }
LABEL_4:
    v38.m128i_i64[0] = (__int64)&unk_49EE2B0;
    if ( v40 != -8 && v40 != 0 && v40 != -16 )
    {
      v30 = v11;
      sub_1649B30(&v38.m128i_u64[1]);
      v11 = v30;
    }
    v35 = &unk_49EE2B0;
    if ( v37 != -8 && v37 != 0 && v37 != -16 )
    {
      v31 = v11;
      sub_1649B30(v36);
      v11 = v31;
    }
    if ( v8 )
    {
      if ( v11 != *(_QWORD *)(a1 + 32) + 48LL * *(unsigned int *)(a1 + 48) )
      {
        v13 = sub_135B4D0(a1, a3);
        if ( !v13[3] )
        {
          v14 = *(unsigned int *)(a1 + 48);
          v15 = *(_QWORD *)(a1 + 32);
          if ( (_DWORD)v14 )
          {
            v27 = *(_DWORD *)(a1 + 48);
            v28 = *(_QWORD *)(a1 + 32);
            v33 = v13;
            sub_1359800(&v35, -8, 0);
            sub_1359800(&v38, -16, 0);
            v13 = v33;
            v20 = (v27 - 1) & v9;
            v16 = v28 + 48LL * v20;
            v21 = *(_QWORD *)(v16 + 24);
            if ( v21 == a2 )
            {
              v22 = v40;
            }
            else
            {
              v22 = v40;
              v25 = v28 + 48LL * v20;
              v26 = 1;
              v16 = 0;
              while ( v37 != v21 )
              {
                if ( !v16 && v40 == v21 )
                  v16 = v25;
                v20 = (v27 - 1) & (v26 + v20);
                v25 = v28 + 48LL * v20;
                v21 = *(_QWORD *)(v25 + 24);
                if ( v21 == a2 )
                {
                  v16 = v28 + 48LL * v20;
                  goto LABEL_21;
                }
                ++v26;
              }
              v8 = 0;
              if ( !v16 )
                v16 = v25;
            }
LABEL_21:
            v38.m128i_i64[0] = (__int64)&unk_49EE2B0;
            if ( v22 != 0 && v22 != -8 && v22 != -16 )
            {
              sub_1649B30(&v38.m128i_u64[1]);
              v13 = v33;
            }
            v35 = &unk_49EE2B0;
            if ( v37 != 0 && v37 != -8 && v37 != -16 )
            {
              v34 = v13;
              sub_1649B30(v36);
              v13 = v34;
            }
            if ( v8 )
              goto LABEL_15;
            v15 = *(_QWORD *)(a1 + 32);
            v14 = *(unsigned int *)(a1 + 48);
          }
          v16 = v15 + 48 * v14;
LABEL_15:
          v32 = v13;
          v17 = sub_13582D0(*(_QWORD *)(v16 + 40), a1);
          v18 = *(_QWORD *)(v16 + 40);
          v19 = *(_QWORD *)(v18 + 40);
          if ( v19 != -8 && v19 != -16 || *(_QWORD *)(v18 + 48) || *(_QWORD *)(v18 + 56) )
          {
            v38 = _mm_loadu_si128((const __m128i *)(v18 + 40));
            v39 = *(_QWORD *)(v18 + 56);
          }
          else
          {
            v38 = 0u;
            v39 = 0;
          }
          sub_13585C0(v17, a1, v32, *(_QWORD *)(v18 + 32), &v38, 1);
        }
      }
    }
  }
}
