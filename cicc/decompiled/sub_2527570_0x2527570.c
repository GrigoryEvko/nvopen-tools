// Function: sub_2527570
// Address: 0x2527570
//
_BYTE *__fastcall sub_2527570(__int64 a1, __m128i *a2, __int64 a3, _BYTE *a4)
{
  int v8; // edx
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdi
  int v13; // r11d
  unsigned int i; // eax
  __int64 v15; // rsi
  unsigned int v16; // eax
  __int64 v17; // r8
  _BYTE *v18; // rax
  __int64 v20; // rdx
  _BYTE *v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // rbx
  _BYTE *v24; // r12
  void (__fastcall *v25)(_BYTE *, _BYTE *, __int64); // rax
  _BYTE *v26; // rax
  _BYTE *v27; // [rsp+20h] [rbp-90h] BYREF
  __int64 v28; // [rsp+28h] [rbp-88h]
  _BYTE *v29; // [rsp+30h] [rbp-80h]
  __int64 v30; // [rsp+38h] [rbp-78h]
  _BYTE *v31; // [rsp+40h] [rbp-70h] BYREF
  __int64 v32; // [rsp+48h] [rbp-68h]
  _BYTE v33[96]; // [rsp+50h] [rbp-60h] BYREF

  v8 = *(_DWORD *)(a1 + 56);
  v9 = *(_QWORD *)(a1 + 40);
  if ( !v8 )
    goto LABEL_8;
  v10 = a2->m128i_i64[0];
  v11 = (unsigned int)(v8 - 1);
  v12 = a2->m128i_i64[1];
  v13 = 1;
  for ( i = v11
          & (((unsigned int)v12 >> 9)
           ^ ((unsigned int)v12 >> 4)
           ^ (16 * (((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)))); ; i = v11 & v16 )
  {
    v15 = v9 + ((unsigned __int64)i << 6);
    if ( v10 == *(_QWORD *)v15 && v12 == *(_QWORD *)(v15 + 8) )
      break;
    if ( unk_4FEE4D0 == *(_QWORD *)v15 && unk_4FEE4D8 == *(_QWORD *)(v15 + 8) )
      goto LABEL_8;
    v16 = v13 + i;
    ++v13;
  }
  v31 = v33;
  v32 = 0x100000000LL;
  v17 = *(unsigned int *)(v15 + 24);
  if ( !(_DWORD)v17 )
    goto LABEL_8;
  sub_2511E10((__int64)&v31, (__int64 *)(v15 + 16), v11, v10, v17);
  if ( !(_DWORD)v32 )
  {
    if ( v31 != v33 )
    {
      _libc_free((unsigned __int64)v31);
      v18 = (_BYTE *)sub_250D070(a2);
      if ( *v18 <= 0x15u )
        goto LABEL_9;
LABEL_28:
      v31 = v33;
      v32 = 0x300000000LL;
      if ( !(unsigned __int8)sub_2526B50(a1, a2, a3, (__int64)&v31, 2u, a4, 1u) )
        goto LABEL_32;
      if ( !(_DWORD)v32 )
      {
        LOBYTE(v28) = 0;
        goto LABEL_33;
      }
      v26 = (_BYTE *)sub_2554630(a1, a3, a2, &v31);
      if ( v26 && *v26 <= 0x15u )
      {
        v27 = v26;
        LOBYTE(v28) = 1;
      }
      else
      {
LABEL_32:
        v27 = 0;
        LOBYTE(v28) = 1;
      }
LABEL_33:
      if ( v31 != v33 )
        _libc_free((unsigned __int64)v31);
      return v27;
    }
LABEL_8:
    v18 = (_BYTE *)sub_250D070(a2);
    if ( *v18 <= 0x15u )
    {
LABEL_9:
      v27 = v18;
      LOBYTE(v28) = 1;
      return v27;
    }
    goto LABEL_28;
  }
  v27 = (_BYTE *)a3;
  if ( !*((_QWORD *)v31 + 2) )
    sub_4263D6(v31, (unsigned int)v32, v20);
  v21 = (_BYTE *)(*((__int64 (__fastcall **)(_BYTE *, __m128i *, _BYTE **, _BYTE *))v31 + 3))(v31, a2, &v27, a4);
  v30 = v22;
  v29 = v21;
  if ( (_BYTE)v22 )
  {
    if ( v21 && *v21 <= 0x15u )
    {
      v27 = v21;
      LOBYTE(v28) = 1;
    }
    else
    {
      v27 = 0;
      LOBYTE(v28) = 1;
    }
  }
  else
  {
    LOBYTE(v28) = 0;
  }
  v23 = (unsigned __int64)v31;
  v24 = &v31[32 * (unsigned int)v32];
  if ( v31 != v24 )
  {
    do
    {
      v25 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v24 - 2);
      v24 -= 32;
      if ( v25 )
        v25(v24, v24, 3);
    }
    while ( (_BYTE *)v23 != v24 );
    v24 = v31;
  }
  if ( v24 != v33 )
    _libc_free((unsigned __int64)v24);
  return v27;
}
