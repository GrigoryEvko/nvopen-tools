// Function: sub_39439C0
// Address: 0x39439c0
//
__int64 __fastcall sub_39439C0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned int v8; // r13d
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 **v12; // rsi
  unsigned __int64 v13; // rax
  int v14; // ebx
  int v15; // r14d
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // rdi
  __m128i si128; // xmm0
  char *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  _WORD *v30; // rdx
  __int64 v31; // rdi
  __int64 *v32[2]; // [rsp+10h] [rbp-90h] BYREF
  _BYTE v33[128]; // [rsp+20h] [rbp-80h] BYREF

  result = sub_393F6A0((__int64)a1, 2885681152LL, a3, a4, a5, a6);
  if ( !(_DWORD)result )
  {
    v8 = 0;
    v9 = a1[9];
    v10 = a1[10];
    v11 = *(_QWORD *)(v9 + 8);
    v12 = (__int64 **)(v10 + 4);
    v13 = *(_QWORD *)(v9 + 16) - v11;
    if ( v13 < v10 + 4 )
    {
      v21 = sub_16E8CB0();
      v22 = (__m128i *)v21[3];
      v23 = (__int64)v21;
      if ( v21[2] - (_QWORD)v22 <= 0x20u )
      {
        v23 = sub_16E7EE0((__int64)v21, "Unexpected end of memory buffer: ", 0x21u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4530950);
        v22[2].m128i_i8[0] = 32;
        *v22 = si128;
        v22[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
        v21[3] += 33LL;
      }
      v25 = (char *)(a1[10] + 4LL);
      v26 = sub_16E7A90(v23, (__int64)v25);
      v30 = *(_WORD **)(v26 + 24);
      v31 = v26;
      if ( *(_QWORD *)(v26 + 16) - (_QWORD)v30 <= 1u )
      {
        v25 = ".\n";
        sub_16E7EE0(v26, ".\n", 2u);
      }
      else
      {
        *v30 = 2606;
        *(_QWORD *)(v26 + 24) += 2LL;
      }
      v8 = 4;
      sub_393D180(v31, (__int64)v25, (__int64)v30, v27, v28, v29);
    }
    else
    {
      a1[10] = v12;
      if ( v13 > v10 )
        v13 = v10;
      v14 = *(_DWORD *)(v11 + v13);
      v32[0] = (__int64 *)v33;
      v32[1] = (__int64 *)0xA00000000LL;
      if ( v14 )
      {
        v15 = 0;
        while ( 1 )
        {
          v12 = v32;
          v16 = sub_3942D70(a1, v32, 1u, 0);
          if ( v16 )
            break;
          if ( ++v15 == v14 )
            goto LABEL_13;
        }
        v8 = v16;
      }
      else
      {
LABEL_13:
        sub_393FC70((__int64)a1);
        sub_393D180((__int64)a1, (__int64)v12, v17, v18, v19, v20);
      }
      if ( (_BYTE *)v32[0] != v33 )
        _libc_free((unsigned __int64)v32[0]);
    }
    return v8;
  }
  return result;
}
