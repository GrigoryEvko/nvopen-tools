// Function: sub_21DF9F0
// Address: 0x21df9f0
//
__int64 __fastcall sub_21DF9F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  _QWORD *v4; // r15
  __int64 v5; // rcx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  int v8; // eax
  __int64 v9; // r9
  const __m128i *v10; // r13
  __int64 v11; // rbx
  __int64 *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v18; // [rsp+8h] [rbp-A8h]
  unsigned int v19; // [rsp+14h] [rbp-9Ch]
  __int64 v20; // [rsp+20h] [rbp-90h] BYREF
  int v21; // [rsp+28h] [rbp-88h]
  __int64 *v22; // [rsp+30h] [rbp-80h] BYREF
  __int64 v23; // [rsp+38h] [rbp-78h]
  _BYTE v24[112]; // [rsp+40h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD **)(a1 - 176);
  v20 = v3;
  if ( v3 )
    sub_1623A60((__int64)&v20, v3, 2);
  v5 = *(_QWORD *)(a2 + 32);
  v21 = *(_DWORD *)(a2 + 64);
  v6 = *(_QWORD *)(*(_QWORD *)v5 + 88LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v22 = (__int64 *)v24;
  v23 = 0x400000000LL;
  if ( (_DWORD)v7 == 4208 )
  {
    v8 = 2;
    v9 = 3219;
  }
  else if ( (unsigned int)v7 > 0x1070 )
  {
    v16 = 0;
    if ( (_DWORD)v7 != 4516 )
      goto LABEL_19;
    v8 = 4;
    v9 = 3636;
  }
  else if ( (_DWORD)v7 == 3658 )
  {
    v8 = 4;
    v9 = 135;
  }
  else
  {
    if ( (_DWORD)v7 != 4109 )
    {
      v16 = 0;
      goto LABEL_19;
    }
    v8 = 3;
    v9 = 3136;
  }
  v10 = (const __m128i *)(v5 + 40);
  v11 = 80;
  v12 = (__int64 *)v24;
  v13 = 5LL * (unsigned int)(v8 + 1);
  v14 = 0;
  v15 = 8 * v13;
  while ( 1 )
  {
    *(__m128i *)&v12[2 * v14] = _mm_loadu_si128(v10);
    v14 = (unsigned int)(v23 + 1);
    LODWORD(v23) = v23 + 1;
    if ( v15 == v11 )
      break;
    v10 = (const __m128i *)(v11 + *(_QWORD *)(a2 + 32));
    if ( HIDWORD(v23) <= (unsigned int)v14 )
    {
      v18 = v15;
      v19 = v9;
      sub_16CD150((__int64)&v22, v24, 0, 16, v15, v9);
      v14 = (unsigned int)v23;
      v15 = v18;
      v9 = v19;
    }
    v12 = v22;
    v11 += 40;
  }
  v16 = sub_1D23DE0(v4, v9, (__int64)&v20, *(_QWORD *)(a2 + 40), *(_DWORD *)(a2 + 60), v9, v22, (unsigned int)v14);
  if ( v22 != (__int64 *)v24 )
    _libc_free((unsigned __int64)v22);
LABEL_19:
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v16;
}
