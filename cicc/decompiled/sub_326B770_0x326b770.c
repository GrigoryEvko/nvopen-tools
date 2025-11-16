// Function: sub_326B770
// Address: 0x326b770
//
__int64 __fastcall sub_326B770(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        _QWORD *a9,
        __int64 a10)
{
  int v10; // r13d
  _QWORD *v12; // r14
  int v13; // r15d
  __int64 v14; // rax
  int v15; // r12d
  __int64 v16; // r9
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int, __int64); // rax
  char *v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int v21; // esi
  __int64 result; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  __m128i v26; // [rsp+0h] [rbp-90h] BYREF
  __int64 v27; // [rsp+18h] [rbp-78h]
  int v28; // [rsp+24h] [rbp-6Ch]
  __int64 v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h]
  __int64 v31; // [rsp+38h] [rbp-58h]
  char v32[8]; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int16 v33; // [rsp+48h] [rbp-48h]

  v10 = a3;
  v12 = a9;
  v29 = a6;
  v13 = a10;
  v27 = a1;
  v28 = a8;
  v14 = *a9;
  v15 = DWORD2(a8);
  v16 = *(_QWORD *)(a10 + 64);
  v31 = a4;
  v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(v14 + 592);
  v30 = a5;
  v26 = _mm_loadu_si128((const __m128i *)&a7);
  if ( v17 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v32, (__int64)a9, v16, a2, a3);
    v19 = v33;
  }
  else
  {
    v19 = (unsigned __int16)v17((__int64)a9, v16, a2, a3);
  }
  switch ( v15 )
  {
    case 2:
    case 3:
    case 10:
    case 11:
    case 18:
    case 19:
      v24 = 1;
      v21 = ((_DWORD)v30 == v28 && v31 == v29) + 281;
      if ( (_WORD)a2 == 1 || (_WORD)a2 && (v24 = (unsigned __int16)a2, v12[(unsigned __int16)a2 + 14]) )
      {
        if ( (*((_BYTE *)v12 + 500 * v24 + v21 + 6414) & 0xFB) == 0 )
          goto LABEL_16;
      }
      v21 = ((_DWORD)v30 == v28 && v31 == v29) + 279;
      if ( (_WORD)v19 == 1 )
      {
        v25 = 1;
      }
      else
      {
        if ( !(_WORD)v19 )
          goto LABEL_6;
        v25 = (unsigned __int16)v19;
        if ( !v12[v19 + 14] )
          goto LABEL_6;
      }
      v18 = (char *)v12 + 500 * v25;
      if ( (v18[v21 + 6414] & 0xFB) != 0 )
        goto LABEL_6;
      goto LABEL_16;
    case 4:
    case 5:
    case 12:
    case 13:
    case 20:
    case 21:
      v23 = 1;
      v21 = ((_DWORD)v30 != v28 || v31 != v29) + 281;
      if ( (_WORD)a2 != 1 && (!(_WORD)a2 || (v23 = (unsigned __int16)a2, !v12[(unsigned __int16)a2 + 14]))
        || (*((_BYTE *)v12 + 500 * v23 + v21 + 6414) & 0xFB) != 0 )
      {
        v20 = 1;
        v21 = ((_DWORD)v30 != v28 || v31 != v29) + 279;
        if ( (_WORD)v19 != 1 )
        {
          if ( !(_WORD)v19 )
            goto LABEL_6;
          v20 = (unsigned __int16)v19;
          if ( !v12[v19 + 14] )
            goto LABEL_6;
        }
        v18 = (char *)v12 + 500 * v20;
        if ( (v18[v21 + 6414] & 0xFB) != 0 )
          goto LABEL_6;
      }
LABEL_16:
      a8 = (__int128)_mm_load_si128(&v26);
      *(_QWORD *)&a7 = v31;
      *((_QWORD *)&a7 + 1) = v30;
      result = sub_3406EB0(v13, v21, v27, a2, v10, (_DWORD)v18, a7, a8);
      break;
    default:
LABEL_6:
      result = 0;
      break;
  }
  return result;
}
