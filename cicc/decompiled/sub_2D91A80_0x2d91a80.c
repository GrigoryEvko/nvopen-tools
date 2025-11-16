// Function: sub_2D91A80
// Address: 0x2d91a80
//
__int64 __fastcall sub_2D91A80(__int64 a1)
{
  unsigned int v1; // r12d
  int v3; // r15d
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rax
  volatile signed __int32 *v7; // rdx
  signed __int32 v8; // eax
  volatile signed __int32 *v9; // r13
  signed __int32 v10; // eax
  signed __int32 v11; // eax
  _QWORD *v12; // rax
  __m128i *v13; // rdx
  __int64 v14; // r12
  __m128i si128; // xmm0
  __m128i v16; // xmm0
  __int64 v17; // rdx
  __int64 (__fastcall **v18)(); // rsi
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 (__fastcall **v21)(); // rax
  signed __int32 v22; // eax
  _QWORD v23[2]; // [rsp+0h] [rbp-A0h] BYREF
  char v24; // [rsp+10h] [rbp-90h]
  __int64 v25[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int8 *v27[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v28[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v29; // [rsp+60h] [rbp-40h]

  sub_2D916A0((__int64 *)v27);
  v1 = sub_2241AC0((__int64)v27, "all");
  if ( (_QWORD *)v27[0] != v28 )
    j_j___libc_free_0((unsigned __int64)v27[0]);
  if ( v1 )
  {
    sub_2D916A0((__int64 *)v27);
    v3 = sub_2241AC0((__int64)v27, "none");
    if ( (_QWORD *)v27[0] != v28 )
      j_j___libc_free_0((unsigned __int64)v27[0]);
    v1 = 3;
    if ( v3 )
    {
      sub_2D916A0(v25);
      v29 = 260;
      v27[0] = (unsigned __int8 *)v25;
      sub_C7EA90((__int64)v23, (__int64 *)v27, 0, 1u, 0, 0);
      if ( (__int64 *)v25[0] != &v26 )
        j_j___libc_free_0(v25[0]);
      if ( (v24 & 1) != 0 )
      {
        v12 = sub_CB72A0();
        v13 = (__m128i *)v12[4];
        v14 = (__int64)v12;
        if ( v12[3] - (_QWORD)v13 <= 0x36u )
        {
          v14 = sub_CB6200((__int64)v12, "Error loading basic block sections function list file: ", 0x37u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_444E8C0);
          v13[3].m128i_i32[0] = 1818846752;
          v13[3].m128i_i16[2] = 14949;
          *v13 = si128;
          v16 = _mm_load_si128((const __m128i *)&xmmword_444E8D0);
          v13[3].m128i_i8[6] = 32;
          v13[1] = v16;
          v13[2] = _mm_load_si128((const __m128i *)&xmmword_444E8E0);
          v12[4] += 55LL;
        }
        if ( (v24 & 1) != 0 )
        {
          v17 = LODWORD(v23[0]);
          v18 = (__int64 (__fastcall **)())v23[1];
        }
        else
        {
          v21 = sub_2241E40();
          v17 = 0;
          v18 = v21;
        }
        (*((void (__fastcall **)(unsigned __int8 **, __int64 (__fastcall **)(), __int64))*v18 + 4))(v27, v18, v17);
        v19 = sub_CB6200(v14, v27[0], (size_t)v27[1]);
        v20 = *(_BYTE **)(v19 + 32);
        if ( *(_BYTE **)(v19 + 24) == v20 )
        {
          sub_CB6200(v19, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v20 = 10;
          ++*(_QWORD *)(v19 + 32);
        }
        if ( (_QWORD *)v27[0] != v28 )
          j_j___libc_free_0((unsigned __int64)v27[0]);
      }
      else
      {
        v4 = v23[0];
        if ( v23[0] && (v5 = sub_22077B0(0x18u), v6 = v23[0], v23[0] = 0, v5) )
        {
          *(_QWORD *)(v5 + 16) = v6;
          *(_QWORD *)(v5 + 8) = 0x100000001LL;
          *(_QWORD *)v5 = &unk_49E5010;
          v7 = (volatile signed __int32 *)(v5 + 8);
          if ( &_pthread_key_create )
          {
            _InterlockedAdd(v7, 1u);
            v8 = _InterlockedExchangeAdd(v7, 0xFFFFFFFF);
          }
          else
          {
            v8 = ++*(_DWORD *)(v5 + 8);
            *(_DWORD *)(v5 + 8) = v8 - 1;
          }
          if ( v8 == 1 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 16LL))(v5);
            if ( &_pthread_key_create )
            {
              v22 = _InterlockedExchangeAdd((volatile signed __int32 *)(v5 + 12), 0xFFFFFFFF);
            }
            else
            {
              v22 = *(_DWORD *)(v5 + 12);
              *(_DWORD *)(v5 + 12) = v22 - 1;
            }
            if ( v22 == 1 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 24LL))(v5);
          }
        }
        else
        {
          v5 = 0;
        }
        *(_QWORD *)(a1 + 32) = v4;
        v9 = *(volatile signed __int32 **)(a1 + 40);
        *(_QWORD *)(a1 + 40) = v5;
        if ( v9 )
        {
          if ( &_pthread_key_create )
          {
            v10 = _InterlockedExchangeAdd(v9 + 2, 0xFFFFFFFF);
          }
          else
          {
            v10 = *((_DWORD *)v9 + 2);
            *((_DWORD *)v9 + 2) = v10 - 1;
          }
          if ( v10 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v9 + 16LL))(v9);
            if ( &_pthread_key_create )
            {
              v11 = _InterlockedExchangeAdd(v9 + 3, 0xFFFFFFFF);
            }
            else
            {
              v11 = *((_DWORD *)v9 + 3);
              *((_DWORD *)v9 + 3) = v11 - 1;
            }
            if ( v11 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v9 + 24LL))(v9);
          }
        }
      }
      if ( (v24 & 1) == 0 && v23[0] )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v23[0] + 8LL))(v23[0]);
      return 1;
    }
  }
  return v1;
}
