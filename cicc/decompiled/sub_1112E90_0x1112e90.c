// Function: sub_1112E90
// Address: 0x1112e90
//
__int64 __fastcall sub_1112E90(const __m128i *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 *v7; // rax
  __int64 v8; // r14
  __int64 v9; // rdx
  bool v10; // r13
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __m128i v13; // xmm3
  __int64 v14; // rax
  __int64 v15; // r8
  char v16; // al
  __int64 v17; // r15
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // r9
  __int64 v21; // r15
  __int64 v22; // r14
  __int64 v23; // r15
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // rbx
  __int64 v28; // r14
  __int64 v29; // rdx
  unsigned int v30; // esi
  unsigned int v31; // r12d
  bool v32; // al
  __int64 v33; // r12
  _BYTE *v34; // rax
  unsigned int v35; // r12d
  _BYTE *v36; // rax
  __int64 v37; // [rsp+8h] [rbp-D8h]
  __int64 v38; // [rsp+18h] [rbp-C8h]
  __int64 v39; // [rsp+20h] [rbp-C0h]
  int v40; // [rsp+20h] [rbp-C0h]
  __int16 v41; // [rsp+28h] [rbp-B8h]
  _BYTE v42[32]; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v43; // [rsp+50h] [rbp-90h]
  _OWORD v44[2]; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v45; // [rsp+80h] [rbp-60h]
  __int64 v46; // [rsp+88h] [rbp-58h]
  __m128i v47; // [rsp+90h] [rbp-50h]
  __int64 v48; // [rsp+A0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 - 64);
  v4 = *(_QWORD *)(v3 + 16);
  if ( !v4 )
    return 0;
  if ( *(_QWORD *)(v4 + 8) )
    return 0;
  if ( (unsigned __int8)(*(_BYTE *)v3 - 51) > 1u )
    return 0;
  v7 = (__int64 *)sub_986520(v3);
  v37 = *v7;
  if ( !*v7 )
    return 0;
  v8 = v7[4];
  if ( !v8 )
    return 0;
  v38 = *(_QWORD *)(a2 - 32);
  if ( !v38 )
    BUG();
  if ( *(_BYTE *)v38 > 0x15u )
    return 0;
  v10 = sub_AC30F0(v38);
  if ( v10 )
    goto LABEL_11;
  if ( *(_BYTE *)v38 != 17 )
  {
    v33 = *(_QWORD *)(v38 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17 <= 1 )
    {
      v34 = sub_AD7630(v38, 0, v9);
      if ( v34 && *v34 == 17 )
      {
        v10 = sub_9867B0((__int64)(v34 + 24));
LABEL_31:
        if ( v10 )
          goto LABEL_11;
      }
      else if ( *(_BYTE *)(v33 + 8) == 17 )
      {
        v40 = *(_DWORD *)(v33 + 32);
        if ( v40 )
        {
          v35 = 0;
          while ( 1 )
          {
            v36 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v38, v35);
            if ( !v36 )
              break;
            if ( *v36 != 13 )
            {
              if ( *v36 != 17 )
                break;
              v10 = sub_9867B0((__int64)(v36 + 24));
              if ( !v10 )
                break;
            }
            if ( v40 == ++v35 )
              goto LABEL_31;
          }
        }
      }
    }
    return 0;
  }
  v31 = *(_DWORD *)(v38 + 32);
  if ( v31 <= 0x40 )
    v32 = *(_QWORD *)(v38 + 24) == 0;
  else
    v32 = v31 == (unsigned int)sub_C444A0(v38 + 24);
  if ( !v32 )
    return 0;
LABEL_11:
  v41 = sub_B53900(a2);
  v11 = _mm_loadu_si128(a1 + 6);
  v12 = _mm_loadu_si128(a1 + 7);
  v13 = _mm_loadu_si128(a1 + 9);
  v14 = a1[10].m128i_i64[0];
  v45 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v48 = v14;
  v46 = a2;
  v44[0] = v11;
  v44[1] = v12;
  v47 = v13;
  v16 = sub_9A1DB0((unsigned __int8 *)v8, 1, 0, (__int64)v44, v15);
  v5 = 0;
  if ( v16 )
  {
    v17 = a1[2].m128i_i64[0];
    v43 = 257;
    v39 = sub_AD62B0(*(_QWORD *)(v8 + 8));
    v18 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v17 + 80) + 32LL))(
            *(_QWORD *)(v17 + 80),
            13,
            v8,
            v39,
            0,
            0);
    if ( !v18 )
    {
      LOWORD(v45) = 257;
      v18 = sub_B504D0(13, v8, v39, (__int64)v44, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v17 + 88) + 16LL))(
        *(_QWORD *)(v17 + 88),
        v18,
        v42,
        *(_QWORD *)(v17 + 56),
        *(_QWORD *)(v17 + 64));
      v22 = 16LL * *(unsigned int *)(v17 + 8);
      v23 = *(_QWORD *)v17;
      v24 = v23 + v22;
      while ( v24 != v23 )
      {
        v25 = *(_QWORD *)(v23 + 8);
        v26 = *(_DWORD *)v23;
        v23 += 16;
        sub_B99FD0(v18, v26, v25);
      }
    }
    v19 = a1[2].m128i_i64[0];
    v43 = 257;
    v21 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v19 + 80) + 16LL))(
            *(_QWORD *)(v19 + 80),
            28,
            v37,
            v18);
    if ( !v21 )
    {
      LOWORD(v45) = 257;
      v21 = sub_B504D0(28, v37, v18, (__int64)v44, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v19 + 88) + 16LL))(
        *(_QWORD *)(v19 + 88),
        v21,
        v42,
        *(_QWORD *)(v19 + 56),
        *(_QWORD *)(v19 + 64));
      v27 = *(_QWORD *)v19;
      v28 = *(_QWORD *)v19 + 16LL * *(unsigned int *)(v19 + 8);
      while ( v28 != v27 )
      {
        v29 = *(_QWORD *)(v27 + 8);
        v30 = *(_DWORD *)v27;
        v27 += 16;
        sub_B99FD0(v21, v30, v29);
      }
    }
    LOWORD(v45) = 257;
    return sub_B52500(53, v41, v21, v38, (__int64)v44, v20, 0, 0);
  }
  return v5;
}
