// Function: sub_373A8D0
// Address: 0x373a8d0
//
void __fastcall sub_373A8D0(__int64 *a1)
{
  __int64 v1; // rbx
  _QWORD *v2; // rcx
  char v3; // al
  __int64 v4; // r8
  int v5; // eax
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  _QWORD *v8; // rdx
  int v9; // eax
  __m128i v10; // rax
  __m128i v11; // xmm1
  __int64 v12; // [rsp+8h] [rbp-118h]
  __int64 v13; // [rsp+10h] [rbp-110h]
  __m128i v14; // [rsp+20h] [rbp-100h] BYREF
  __m128i v15; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v16; // [rsp+40h] [rbp-E0h]
  _QWORD v17[4]; // [rsp+50h] [rbp-D0h] BYREF
  __int16 v18; // [rsp+70h] [rbp-B0h]
  __m128i v19; // [rsp+80h] [rbp-A0h] BYREF
  __m128i v20; // [rsp+90h] [rbp-90h]
  __int64 v21; // [rsp+A0h] [rbp-80h]
  _BYTE *v22; // [rsp+B0h] [rbp-70h] BYREF
  size_t v23; // [rsp+B8h] [rbp-68h]
  __int64 v24; // [rsp+C0h] [rbp-60h]
  _BYTE v25[88]; // [rsp+C8h] [rbp-58h] BYREF

  v1 = a1[96];
  v12 = a1[95];
  if ( v12 != v1 )
  {
    while ( 1 )
    {
      v1 -= 16;
      v6 = sub_A777F0(0x30u, a1 + 11);
      if ( v6 )
      {
        *(_BYTE *)(v6 + 30) = 0;
        v7 = v6;
        *(_QWORD *)(v6 + 8) = 0;
        *(_QWORD *)v6 = v6 | 4;
        *(_QWORD *)(v6 + 16) = 0;
        *(_DWORD *)(v6 + 24) = -1;
        *(_WORD *)(v6 + 28) = 36;
        *(_QWORD *)(v6 + 32) = 0;
      }
      else
      {
        v7 = 0;
      }
      *(_QWORD *)(v6 + 40) = (unsigned __int64)(a1 + 1) & 0xFFFFFFFFFFFFFFFBLL;
      v8 = (_QWORD *)a1[5];
      if ( !v8 )
        break;
      *(_QWORD *)v6 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)a1[5] = v7 | 4;
      v7 = a1[5];
      if ( v7 )
        goto LABEL_14;
LABEL_15:
      v22 = v25;
      v23 = 0;
      v24 = 32;
      v9 = *(_DWORD *)v1;
      v18 = 265;
      LODWORD(v17[0]) = v9;
      v10.m128i_i64[0] = (__int64)sub_E09D50(*(unsigned __int8 *)(v1 + 4));
      v14 = v10;
      v15.m128i_i64[0] = (__int64)"_";
      v3 = v18;
      LOWORD(v16) = 773;
      if ( (_BYTE)v18 )
      {
        if ( (_BYTE)v18 == 1 )
        {
          v11 = _mm_loadu_si128(&v15);
          v19 = _mm_loadu_si128(&v14);
          v21 = v16;
          v20 = v11;
        }
        else
        {
          if ( HIBYTE(v18) == 1 )
          {
            v13 = v17[1];
            v2 = (_QWORD *)v17[0];
          }
          else
          {
            v2 = v17;
            v3 = 2;
          }
          v20.m128i_i64[0] = (__int64)v2;
          v19.m128i_i64[0] = (__int64)&v14;
          v20.m128i_i64[1] = v13;
          LOBYTE(v21) = 2;
          BYTE1(v21) = v3;
        }
      }
      else
      {
        LOWORD(v21) = 256;
      }
      sub_CA0EC0((__int64)&v19, (__int64)&v22);
      sub_324AD70(a1, v7, 3, v22, v23);
      v4 = *(unsigned __int8 *)(v1 + 4);
      v19.m128i_i32[0] = 65547;
      sub_3249A20(a1, (unsigned __int64 **)(v7 + 8), 62, 65547, v4);
      v5 = *(_DWORD *)v1;
      v19.m128i_i8[2] = 0;
      sub_3249A20(
        a1,
        (unsigned __int64 **)(v7 + 8),
        11,
        v19.m128i_i32[0],
        (((v5 - (unsigned int)(v5 != 0)) >> 3) + (v5 != 0)) & 0x3FFFFFFF);
      *(_QWORD *)(v1 + 8) = v7;
      if ( v22 != v25 )
        _libc_free((unsigned __int64)v22);
      if ( v12 == v1 )
        return;
    }
    a1[5] = v7;
LABEL_14:
    v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_15;
  }
}
