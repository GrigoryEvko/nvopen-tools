// Function: sub_1CF6140
// Address: 0x1cf6140
//
unsigned __int64 __fastcall sub_1CF6140(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // r12
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // r12
  __m128i *v14; // rsi
  __int64 v15; // rcx
  __m128i *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rax
  _DWORD *v21; // rdi
  __m128i *v22; // rsi
  __m128i v23[3]; // [rsp+0h] [rbp-A0h] BYREF
  __m128i v24; // [rsp+30h] [rbp-70h] BYREF
  __m128i v25; // [rsp+40h] [rbp-60h] BYREF
  __m128i v26; // [rsp+50h] [rbp-50h] BYREF
  __m128i v27; // [rsp+60h] [rbp-40h] BYREF
  char v28; // [rsp+70h] [rbp-30h]

  result = (unsigned int)*(unsigned __int8 *)(a2 + 16) - 24;
  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1D:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x48:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x4F:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x58:
      goto LABEL_2;
    case 0x36:
      result = sub_1CF56E0((__int64)&v24, (_QWORD *)a1, a2, a4);
      if ( v28 )
      {
        v14 = *(__m128i **)(a1 + 136);
        if ( v14 == *(__m128i **)(a1 + 144) )
        {
          return (unsigned __int64)sub_1CF5F80((const __m128i **)(a1 + 128), v14, &v24);
        }
        else
        {
          if ( v14 )
          {
            *v14 = _mm_loadu_si128(&v24);
            v14[1] = _mm_loadu_si128(&v25);
            v14[2] = _mm_loadu_si128(&v26);
            v14[3] = _mm_loadu_si128(&v27);
            v14 = *(__m128i **)(a1 + 136);
          }
          *(_QWORD *)(a1 + 136) = v14 + 4;
        }
      }
      return result;
    case 0x37:
      if ( !(unsigned int)sub_1648720(*(_QWORD *)(a1 + 336)) )
        goto LABEL_2;
      if ( *(_QWORD *)(a1 + 48) )
        goto LABEL_27;
      v17 = sub_1CF54D0(a1, a2);
      v18 = v17;
      if ( !v17 )
        goto LABEL_27;
      v19 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL);
      v20 = *(_QWORD *)(*(_QWORD *)v17 + 24LL);
      if ( *(_BYTE *)(v20 + 8) != 13 || *(_BYTE *)(v19 + 8) != 13 )
      {
        if ( v19 != v20 )
          goto LABEL_27;
        goto LABEL_41;
      }
      if ( *(_DWORD *)(v20 + 12) )
      {
        if ( *(_DWORD *)(v19 + 12) )
        {
          v15 = *(_QWORD *)(v20 + 16);
          if ( *(_QWORD *)v15 == **(_QWORD **)(v19 + 16) )
          {
LABEL_41:
            *(_QWORD *)(a1 + 48) = a2;
            sub_141EDF0(v23, a2);
            v21 = (_DWORD *)(a1 + 56);
            v15 = 10;
            v22 = v23;
            while ( v15 )
            {
              *v21 = v22->m128i_i32[0];
              v22 = (__m128i *)((char *)v22 + 4);
              ++v21;
              --v15;
            }
            *(_QWORD *)(a1 + 96) = v18;
          }
        }
      }
LABEL_27:
      result = sub_1CF56E0((__int64)&v24, (_QWORD *)a1, a2, v15);
      if ( v28 )
      {
        v16 = *(__m128i **)(a1 + 112);
        if ( v16 == *(__m128i **)(a1 + 120) )
        {
          return (unsigned __int64)sub_1CF5F80((const __m128i **)(a1 + 104), v16, &v24);
        }
        else
        {
          if ( v16 )
          {
            *v16 = _mm_loadu_si128(&v24);
            v16[1] = _mm_loadu_si128(&v25);
            v16[2] = _mm_loadu_si128(&v26);
            v16[3] = _mm_loadu_si128(&v27);
          }
          *(_QWORD *)(a1 + 112) += 64LL;
        }
      }
      return result;
    case 0x38:
      result = sub_1648720(*(_QWORD *)(a1 + 336));
      if ( (_DWORD)result )
      {
LABEL_2:
        result = *(_QWORD *)(a1 + 336);
        *(_QWORD *)(a1 + 328) = result;
      }
      else
      {
        v13 = *(_QWORD *)(a2 + 8);
        if ( v13 )
        {
          result = *(unsigned int *)(a1 + 160);
          do
          {
            if ( *(_DWORD *)(a1 + 164) <= (unsigned int)result )
            {
              sub_16CD150(a1 + 152, (const void *)(a1 + 168), 0, 8, v11, v12);
              result = *(unsigned int *)(a1 + 160);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * result) = v13;
            result = (unsigned int)(*(_DWORD *)(a1 + 160) + 1);
            *(_DWORD *)(a1 + 160) = result;
            v13 = *(_QWORD *)(v13 + 8);
          }
          while ( v13 );
        }
      }
      return result;
    case 0x47:
      v10 = *(_QWORD *)(a2 + 8);
      if ( v10 )
      {
        result = *(unsigned int *)(a1 + 160);
        do
        {
          if ( *(_DWORD *)(a1 + 164) <= (unsigned int)result )
          {
            sub_16CD150(a1 + 152, (const void *)(a1 + 168), 0, 8, a5, a6);
            result = *(unsigned int *)(a1 + 160);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * result) = v10;
          result = (unsigned int)(*(_DWORD *)(a1 + 160) + 1);
          *(_DWORD *)(a1 + 160) = result;
          v10 = *(_QWORD *)(v10 + 8);
        }
        while ( v10 );
      }
      return result;
    case 0x4D:
      v9 = *(_QWORD *)(a2 + 8);
      if ( v9 )
      {
        result = *(unsigned int *)(a1 + 160);
        do
        {
          if ( *(_DWORD *)(a1 + 164) <= (unsigned int)result )
          {
            sub_16CD150(a1 + 152, (const void *)(a1 + 168), 0, 8, a5, a6);
            result = *(unsigned int *)(a1 + 160);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * result) = v9;
          result = (unsigned int)(*(_DWORD *)(a1 + 160) + 1);
          *(_DWORD *)(a1 + 160) = result;
          v9 = *(_QWORD *)(v9 + 8);
        }
        while ( v9 );
      }
      return result;
    case 0x4E:
      result = *(_QWORD *)(a2 - 24);
      if ( !*(_BYTE *)(result + 16) )
        result = *(unsigned int *)(result + 36);
      *(_QWORD *)(a1 + 328) = *(_QWORD *)(a1 + 336);
      return result;
  }
}
