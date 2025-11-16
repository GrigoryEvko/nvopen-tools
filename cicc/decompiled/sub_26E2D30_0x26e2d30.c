// Function: sub_26E2D30
// Address: 0x26e2d30
//
void __fastcall sub_26E2D30(__int64 a1, _QWORD *a2, __int64 a3, unsigned __int64 *a4)
{
  __int64 v4; // r15
  __int32 v6; // r14d
  int v7; // r12d
  _DWORD *v8; // rax
  __int64 v9; // r8
  __int64 v10; // rbx
  int v11; // eax
  unsigned __int64 v12; // rdi
  int v13; // ebx
  unsigned __int64 v14; // r12
  int *v15; // rax
  int v16; // edx
  __int32 v17; // esi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  int v21; // eax
  _QWORD *v22; // rdx
  __int64 v23; // rbx
  int v25; // [rsp+24h] [rbp-8Ch]
  __int64 v26; // [rsp+28h] [rbp-88h]
  __m128i v27; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v28; // [rsp+40h] [rbp-70h] BYREF
  __int64 v29; // [rsp+48h] [rbp-68h]
  _BYTE v30[96]; // [rsp+50h] [rbp-60h] BYREF

  v28 = v30;
  v4 = *(_QWORD *)(a3 + 24);
  v29 = 0x600000000LL;
  v26 = a3 + 8;
  v25 = 0;
  if ( v4 != a3 + 8 )
  {
    do
    {
      v6 = *(_DWORD *)(v4 + 36);
      v7 = *(_DWORD *)(v4 + 32);
      v8 = sub_26E2B00(a2, *(_QWORD *)(v4 + 32) % a2[1], (_DWORD *)(v4 + 32), *(_QWORD *)(v4 + 32));
      if ( v8 && (v10 = *(_QWORD *)v8) != 0 )
      {
        v11 = *(_DWORD *)(v10 + 16);
        if ( v7 != v11 || v6 != *(_DWORD *)(v10 + 20) )
        {
          v27.m128i_i64[0] = *(_QWORD *)(v4 + 32);
          v27.m128i_i64[1] = *(_QWORD *)(v10 + 16);
          sub_26E2C70(a4, &v27);
          v11 = *(_DWORD *)(v10 + 16);
          v7 = *(_DWORD *)(v4 + 32);
        }
        v12 = (unsigned int)v29;
        v25 = v11 - v7;
        v13 = v11 - v7;
        v14 = ((unsigned __int64)(unsigned int)v29 + 1) >> 1;
        if ( v14 < (unsigned int)v29 )
        {
          do
          {
            v15 = (int *)&v28[8 * v14];
            v16 = *v15;
            v17 = v15[1];
            if ( *v15 != v13 + *v15 )
            {
              v18 = *(_QWORD *)v15;
              v27.m128i_i32[3] = v17;
              v27.m128i_i32[2] = v13 + v16;
              v27.m128i_i64[0] = v18;
              sub_26E2C70(a4, &v27);
              v12 = (unsigned int)v29;
            }
            ++v14;
          }
          while ( v14 < v12 );
        }
        LODWORD(v29) = 0;
      }
      else
      {
        if ( v7 + v25 != v7 )
        {
          v19 = *(_QWORD *)(v4 + 32);
          v27.m128i_i32[2] = v7 + v25;
          v27.m128i_i32[3] = v6;
          v27.m128i_i64[0] = v19;
          sub_26E2C70(a4, &v27);
        }
        v20 = (unsigned int)v29;
        v21 = v29;
        if ( (unsigned int)v29 >= (unsigned __int64)HIDWORD(v29) )
        {
          v23 = *(_QWORD *)(v4 + 32);
          if ( HIDWORD(v29) < (unsigned __int64)(unsigned int)v29 + 1 )
          {
            sub_C8D5F0((__int64)&v28, v30, (unsigned int)v29 + 1LL, 8u, v9, (unsigned int)v29 + 1LL);
            v20 = (unsigned int)v29;
          }
          *(_QWORD *)&v28[8 * v20] = v23;
          LODWORD(v29) = v29 + 1;
        }
        else
        {
          v22 = &v28[8 * (unsigned int)v29];
          if ( v22 )
          {
            *v22 = *(_QWORD *)(v4 + 32);
            v21 = v29;
          }
          LODWORD(v29) = v21 + 1;
        }
      }
      v4 = sub_220EF30(v4);
    }
    while ( v26 != v4 );
    if ( v28 != v30 )
      _libc_free((unsigned __int64)v28);
  }
}
