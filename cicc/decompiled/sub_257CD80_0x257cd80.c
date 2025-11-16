// Function: sub_257CD80
// Address: 0x257cd80
//
void __fastcall sub_257CD80(__int64 a1, __int64 a2)
{
  __m128i *v3; // r12
  unsigned __int64 v4; // r15
  _BYTE *v5; // rbx
  __int64 *v6; // rdi
  int v7; // eax
  unsigned __int64 v8; // rax
  int v9; // eax
  __int64 i; // rbx
  __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rbx
  int j; // r12d
  __int64 v16; // rcx
  __int64 v17; // r15
  unsigned __int64 v18; // rcx
  unsigned __int64 *v19; // rdi
  _BYTE *v20; // r15
  unsigned int v21; // eax
  __int64 *v22; // r9
  __int64 v23; // r10
  int v24; // eax
  _BYTE *v25; // rdx
  unsigned __int64 *v26; // rax
  __int64 v27; // rax
  int v28; // r9d
  int v29; // r11d
  __int64 v30; // rdx
  _QWORD *v31; // [rsp+18h] [rbp-148h]
  __int64 v32; // [rsp+20h] [rbp-140h]
  __int64 v33; // [rsp+30h] [rbp-130h]
  __int64 *v34; // [rsp+48h] [rbp-118h]
  unsigned __int64 v35; // [rsp+48h] [rbp-118h]
  __int64 v36; // [rsp+48h] [rbp-118h]
  __m128i *v37; // [rsp+58h] [rbp-108h] BYREF
  void *v38; // [rsp+60h] [rbp-100h] BYREF
  __int64 v39; // [rsp+68h] [rbp-F8h]
  __int64 v40; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v41; // [rsp+78h] [rbp-E8h]
  __int64 v42; // [rsp+80h] [rbp-E0h]
  __int64 v43; // [rsp+88h] [rbp-D8h]
  unsigned __int64 *v44; // [rsp+90h] [rbp-D0h]
  __int64 v45; // [rsp+98h] [rbp-C8h]
  _BYTE *v46; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+A8h] [rbp-B8h]
  _BYTE v48[48]; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v49; // [rsp+E0h] [rbp-80h] BYREF
  _QWORD v50[6]; // [rsp+F0h] [rbp-70h] BYREF
  __int16 v51; // [rsp+120h] [rbp-40h]

  v3 = (__m128i *)(a1 + 72);
  v34 = (__int64 *)sub_250D070((_QWORD *)(a1 + 72));
  if ( (unsigned int)*(unsigned __int8 *)v34 - 12 > 1 )
  {
    v46 = v48;
    v47 = 0x600000000LL;
    v49.m128i_i32[0] = 93;
    sub_2515D00(a2, v3, v49.m128i_i32, 1, (__int64)&v46, 0);
    v4 = (unsigned __int64)v46;
    v5 = &v46[8 * (unsigned int)v47];
    if ( v5 != v46 )
    {
      do
      {
        v6 = (__int64 *)v4;
        v4 += 8LL;
        v7 = sub_A71E30(v6);
        *(_DWORD *)(a1 + 100) |= v7;
        *(_DWORD *)(a1 + 96) |= v7;
      }
      while ( v5 != (_BYTE *)v4 );
    }
    if ( (unsigned __int8)sub_2509800(v3) != 2 )
    {
      v8 = *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL);
      memset(v50, 0, sizeof(v50));
      v49 = (__m128i)v8;
      v51 = 257;
      v9 = ~(unsigned __int16)sub_9B4030(v34, 1023, 0, &v49) & 0x3FF;
      *(_DWORD *)(a1 + 100) |= v9;
      *(_DWORD *)(a1 + 96) |= v9;
    }
    v35 = sub_2509740(v3);
    if ( v35 )
    {
      v33 = *(_QWORD *)(*(_QWORD *)(a2 + 208) + 120LL);
      if ( v33 )
      {
        v40 = 0;
        v41 = 0;
        v42 = 0;
        v43 = 0;
        v44 = (unsigned __int64 *)&v46;
        v45 = 0;
        for ( i = *(_QWORD *)(sub_250D070(v3) + 16); i; i = *(_QWORD *)(i + 8) )
        {
          v49.m128i_i64[0] = i;
          sub_25789E0((__int64)&v40, v49.m128i_i64);
        }
        sub_257C940(a1, a2, v33, v35, (__int64)&v40, a1 + 88);
        if ( *(_DWORD *)(a1 + 100) != *(_DWORD *)(a1 + 96) )
        {
          v49.m128i_i64[0] = (__int64)v50;
          v49.m128i_i64[1] = 0x400000000LL;
          v37 = &v49;
          sub_2568920(v33, v35, (unsigned __int8 (__fastcall *)(__int64))sub_253B560, (__int64)&v37);
          v31 = (_QWORD *)(v49.m128i_i64[0] + 8LL * v49.m128i_u32[2]);
          if ( (_QWORD *)v49.m128i_i64[0] != v31 )
          {
            v32 = v49.m128i_i64[0];
            do
            {
              v11 = *(_QWORD *)v32;
              v12 = *(_DWORD *)(*(_QWORD *)v32 + 4LL) & 0x7FFFFFF;
              v13 = 32LL * v12;
              if ( (*(_BYTE *)(*(_QWORD *)v32 + 7LL) & 0x40) != 0 )
              {
                v14 = *(_QWORD *)(v11 - 8);
                v36 = v14 + v13;
                if ( v12 == 3 )
                  v14 += 32;
              }
              else
              {
                v36 = *(_QWORD *)v32;
                v14 = v11 - v13;
                v30 = v11 - v13 + 32;
                if ( v12 == 3 )
                  v14 = v30;
              }
              for ( j = 1023; v36 != v14; v14 += 32 )
              {
                v16 = *(_QWORD *)(*(_QWORD *)v14 + 56LL);
                v38 = &unk_4A172B8;
                v39 = 0x3FF00000000LL;
                v17 = (unsigned int)v45;
                if ( v16 )
                  v16 -= 24;
                sub_257C940(a1, a2, v33, v16, (__int64)&v40, (__int64)&v38);
                v18 = (unsigned __int64)v44;
                v19 = &v44[v17];
                v20 = v19 + 1;
                if ( v19 != &v44[(unsigned int)v45] )
                {
                  do
                  {
                    if ( (_DWORD)v43 )
                    {
                      v21 = (v43 - 1) & (((unsigned int)*v19 >> 9) ^ ((unsigned int)*v19 >> 4));
                      v22 = (__int64 *)(v41 + 8LL * v21);
                      v23 = *v22;
                      if ( *v19 == *v22 )
                      {
LABEL_24:
                        *v22 = -8192;
                        v18 = (unsigned __int64)v44;
                        LODWORD(v42) = v42 - 1;
                        ++HIDWORD(v42);
                      }
                      else
                      {
                        v28 = 1;
                        while ( v23 != -4096 )
                        {
                          v29 = v28 + 1;
                          v21 = (v43 - 1) & (v28 + v21);
                          v22 = (__int64 *)(v41 + 8LL * v21);
                          v23 = *v22;
                          if ( *v19 == *v22 )
                            goto LABEL_24;
                          v28 = v29;
                        }
                      }
                    }
                    v24 = v45;
                    v25 = (_BYTE *)(v18 + 8LL * (unsigned int)v45);
                    if ( v25 != v20 )
                    {
                      v26 = (unsigned __int64 *)memmove(v19, v20, v25 - v20);
                      v18 = (unsigned __int64)v44;
                      v19 = v26;
                      v24 = v45;
                    }
                    v27 = (unsigned int)(v24 - 1);
                    LODWORD(v45) = v27;
                  }
                  while ( v19 != (unsigned __int64 *)(v18 + 8 * v27) );
                }
                j &= v39;
              }
              v32 += 8;
              *(_DWORD *)(a1 + 100) |= j;
              *(_DWORD *)(a1 + 96) |= j;
            }
            while ( v31 != (_QWORD *)v32 );
            v31 = (_QWORD *)v49.m128i_i64[0];
          }
          if ( v31 != v50 )
            _libc_free((unsigned __int64)v31);
        }
        if ( v44 != (unsigned __int64 *)&v46 )
          _libc_free((unsigned __int64)v44);
        sub_C7D6A0(v41, 8LL * (unsigned int)v43, 8);
      }
    }
    if ( v46 != v48 )
      _libc_free((unsigned __int64)v46);
  }
  else
  {
    *(_DWORD *)(a1 + 96) = *(_DWORD *)(a1 + 100);
  }
}
