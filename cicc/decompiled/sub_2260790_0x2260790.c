// Function: sub_2260790
// Address: 0x2260790
//
__int64 __fastcall sub_2260790(__int64 a1)
{
  unsigned int v2; // eax
  __int64 result; // rax
  __int64 v4; // rdi
  char v5; // [rsp+7h] [rbp-79h] BYREF
  __int64 v6; // [rsp+8h] [rbp-78h] BYREF
  _QWORD *v7; // [rsp+10h] [rbp-70h] BYREF
  char *v8; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v9[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v10[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 (__fastcall *v11)(const __m128i **, const __m128i *, int); // [rsp+40h] [rbp-40h]
  _QWORD *(__fastcall *v12)(_QWORD *, __int64 **, __int64); // [rsp+48h] [rbp-38h]
  _QWORD v13[6]; // [rsp+50h] [rbp-30h] BYREF

  v10[0] = a1 + 32;
  v10[1] = a1 + 40;
  v6 = a1;
  v12 = sub_225FFC0;
  v5 = 0;
  v11 = sub_225FF10;
  v8 = &v5;
  v7 = v10;
  v9[0] = sub_16D4680;
  v13[0] = v9;
  v13[1] = &v6;
  v13[2] = &v7;
  v9[1] = 0;
  v13[3] = &v8;
  *(_QWORD *)(__readfsqword(0) - 24) = v13;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_16D42A0;
  if ( !&_pthread_key_create )
  {
    v2 = -1;
LABEL_10:
    sub_4264C5(v2);
  }
  v2 = pthread_once((pthread_once_t *)(a1 + 24), init_routine);
  if ( v2 )
    goto LABEL_10;
  if ( v5 )
  {
    v4 = a1 + 16;
    if ( _InterlockedExchange((volatile __int32 *)(a1 + 16), 1) < 0 )
      sub_222D1B0(v4);
  }
  result = (__int64)v11;
  if ( v11 )
    return v11((const __m128i **)v10, (const __m128i *)v10, 3);
  return result;
}
