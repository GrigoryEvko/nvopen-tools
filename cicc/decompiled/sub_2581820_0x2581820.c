// Function: sub_2581820
// Address: 0x2581820
//
__int64 __fastcall sub_2581820(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned int v7; // r8d
  unsigned __int64 v8; // rdi
  unsigned int v9; // eax
  __m128i v10; // rax
  unsigned int v11; // r15d
  char **v12; // rbx
  unsigned __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rbx
  __int64 v16; // r12
  char **v18; // r15
  unsigned int v19; // eax
  bool v20; // cc
  unsigned int v21; // eax
  bool v22; // zf
  const void ***v23; // [rsp+18h] [rbp-1F8h]
  unsigned int v24; // [rsp+24h] [rbp-1ECh]
  char **v25; // [rsp+28h] [rbp-1E8h]
  char v26; // [rsp+3Fh] [rbp-1D1h] BYREF
  __int64 v27; // [rsp+40h] [rbp-1D0h] BYREF
  int v28; // [rsp+48h] [rbp-1C8h]
  __m128i v29; // [rsp+50h] [rbp-1C0h] BYREF
  __int64 v30; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v31; // [rsp+68h] [rbp-1A8h]
  __int64 v32; // [rsp+70h] [rbp-1A0h]
  __int64 v33; // [rsp+78h] [rbp-198h]
  char **v34; // [rsp+80h] [rbp-190h]
  __int64 v35; // [rsp+88h] [rbp-188h]
  _BYTE v36[128]; // [rsp+90h] [rbp-180h] BYREF
  _BYTE v37[256]; // [rsp+110h] [rbp-100h] BYREF

  v23 = (const void ***)(a1 + 88);
  sub_2560F70((__int64)v37, a1 + 88);
  if ( (unsigned __int8)sub_B50700(a3, a1 + 88, v5, v6, v7) )
  {
    v8 = *((_QWORD *)a3 - 4);
    v9 = *(_DWORD *)(*((_QWORD *)a3 + 1) + 8LL);
    v26 = 0;
    v30 = 0;
    v31 = 0;
    v24 = v9 >> 8;
    v34 = (char **)v36;
    v32 = 0;
    v33 = 0;
    v35 = 0x800000000LL;
    v10.m128i_i64[0] = sub_250D2C0(v8, 0);
    v29 = v10;
    if ( (unsigned __int8)sub_2580850(a1, a2, &v29, &v30, &v26, 0) )
    {
      if ( v26 )
      {
        *(_BYTE *)(a1 + 288) = *(_DWORD *)(a1 + 152) == 0;
      }
      else
      {
        v18 = v34;
        v25 = &v34[2 * (unsigned int)v35];
        if ( v25 != v34 )
        {
          v19 = *a3 - 29;
          v20 = v19 <= 0x28;
          if ( *a3 != 69 )
            goto LABEL_37;
LABEL_27:
          sub_C44830((__int64)&v29, v18, v24);
LABEL_29:
          if ( *(_BYTE *)(a1 + 105) )
          {
LABEL_30:
            sub_2575FB0((_DWORD *)(a1 + 112), (const void **)&v29);
            v21 = *(_DWORD *)(a1 + 152);
            if ( v21 < unk_4FEF868 )
              *(_BYTE *)(a1 + 288) &= v21 == 0;
            else
              *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
          }
          if ( v29.m128i_i32[2] > 0x40u && v29.m128i_i64[0] )
            j_j___libc_free_0_0(v29.m128i_u64[0]);
          while ( 1 )
          {
            v18 += 2;
            if ( v25 == v18 )
              break;
            v19 = *a3 - 29;
            v20 = v19 <= 0x28;
            if ( *a3 == 69 )
              goto LABEL_27;
LABEL_37:
            if ( v20 )
            {
              if ( v19 == 38 )
              {
                sub_C44740((__int64)&v29, v18, v24);
              }
              else
              {
                if ( v19 != 39 )
LABEL_47:
                  BUG();
                sub_C449B0((__int64)&v29, (const void **)v18, v24);
              }
              goto LABEL_29;
            }
            if ( v19 != 49 )
              goto LABEL_47;
            v29.m128i_i32[2] = *((_DWORD *)v18 + 2);
            if ( v29.m128i_i32[2] > 0x40u )
            {
              sub_C43780((__int64)&v29, (const void **)v18);
              goto LABEL_29;
            }
            v22 = *(_BYTE *)(a1 + 105) == 0;
            v29.m128i_i64[0] = (__int64)*v18;
            if ( !v22 )
              goto LABEL_30;
          }
        }
      }
      v11 = (unsigned __int8)sub_255BE50((__int64)v37, v23);
    }
    else
    {
      v11 = 0;
      *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
    }
    v12 = v34;
    v13 = (unsigned __int64)&v34[2 * (unsigned int)v35];
    if ( v34 != (char **)v13 )
    {
      do
      {
        v13 -= 16LL;
        if ( *(_DWORD *)(v13 + 8) > 0x40u && *(_QWORD *)v13 )
          j_j___libc_free_0_0(*(_QWORD *)v13);
      }
      while ( v12 != (char **)v13 );
      v13 = (unsigned __int64)v34;
    }
    if ( (_BYTE *)v13 != v36 )
      _libc_free(v13);
    v14 = (unsigned int)v33;
    if ( (_DWORD)v33 )
    {
      v15 = v31;
      v28 = 0;
      v27 = -1;
      v29.m128i_i32[2] = 0;
      v16 = v31 + 16LL * (unsigned int)v33;
      v29.m128i_i64[0] = -2;
      do
      {
        if ( *(_DWORD *)(v15 + 8) > 0x40u && *(_QWORD *)v15 )
          j_j___libc_free_0_0(*(_QWORD *)v15);
        v15 += 16;
      }
      while ( v16 != v15 );
      sub_969240(v29.m128i_i64);
      sub_969240(&v27);
      v14 = (unsigned int)v33;
    }
    sub_C7D6A0(v31, 16 * v14, 8);
  }
  else
  {
    v11 = 0;
    *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
  }
  sub_25485A0((__int64)v37);
  return v11;
}
