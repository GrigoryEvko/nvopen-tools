// Function: sub_36E41D0
// Address: 0x36e41d0
//
__int64 __fastcall sub_36E41D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // eax
  int v13; // r15d
  __m128i v14; // xmm0
  __int64 v15; // rbx
  unsigned __int64 *v16; // rdx
  __int64 v17; // r8
  unsigned __int64 **v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r8
  unsigned int v21; // r15d
  __int64 v22; // r15
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __m128i v27; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v28; // [rsp+10h] [rbp-A0h]
  unsigned __int64 **v29; // [rsp+18h] [rbp-98h]
  __int64 v30; // [rsp+20h] [rbp-90h] BYREF
  int v31; // [rsp+28h] [rbp-88h]
  unsigned __int64 *v32; // [rsp+30h] [rbp-80h] BYREF
  __int64 v33; // [rsp+38h] [rbp-78h]
  _BYTE v34[112]; // [rsp+40h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v30 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v30, v8, 1);
  v9 = *(_QWORD *)(a2 + 40);
  v31 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(*(_QWORD *)v9 + 96LL);
  if ( *(_DWORD *)(v10 + 32) <= 0x40u )
    v11 = *(_QWORD *)(v10 + 24);
  else
    v11 = **(_QWORD **)(v10 + 24);
  v32 = (unsigned __int64 *)v34;
  v33 = 0x400000000LL;
  if ( (_DWORD)v11 == 9175 )
  {
    v12 = 2;
    v13 = 2945;
  }
  else if ( (unsigned int)v11 > 0x23D7 )
  {
    v21 = 0;
    if ( (_DWORD)v11 != 9597 )
      goto LABEL_20;
    v12 = 4;
    v13 = 3728;
  }
  else if ( (_DWORD)v11 == 8164 )
  {
    v12 = 4;
    v13 = 316;
  }
  else
  {
    if ( (_DWORD)v11 != 9004 )
    {
      v21 = 0;
      goto LABEL_20;
    }
    v12 = 3;
    v13 = 2753;
  }
  v14 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v15 = 80;
  v16 = (unsigned __int64 *)v34;
  v17 = 5LL * (unsigned int)(v12 + 1);
  v18 = &v32;
  v19 = 0;
  v20 = 8 * v17;
  while ( 1 )
  {
    *(__m128i *)&v16[2 * v19] = v14;
    v19 = (unsigned int)(v33 + 1);
    LODWORD(v33) = v33 + 1;
    if ( v20 == v15 )
      break;
    v14 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v15));
    if ( v19 + 1 > (unsigned __int64)HIDWORD(v33) )
    {
      v28 = v20;
      v29 = v18;
      v27 = v14;
      sub_C8D5F0((__int64)v18, v34, v19 + 1, 0x10u, v20, a6);
      v19 = (unsigned int)v33;
      v20 = v28;
      v14 = _mm_load_si128(&v27);
      v18 = v29;
    }
    v16 = v32;
    v15 += 40;
  }
  v22 = sub_33E66D0(
          *(_QWORD **)(a1 + 64),
          v13,
          (__int64)&v30,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          a6,
          v32,
          (unsigned int)v19);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v22, v23, v24, v25);
  sub_3421DB0(v22);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v32 != (unsigned __int64 *)v34 )
    _libc_free((unsigned __int64)v32);
  v21 = 1;
LABEL_20:
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v21;
}
