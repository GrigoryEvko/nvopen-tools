// Function: sub_16D44A0
// Address: 0x16d44a0
//
__int64 __fastcall sub_16D44A0(__int64 a1)
{
  unsigned int v1; // eax
  __int64 result; // rax
  char v3; // [rsp+Fh] [rbp-81h] BYREF
  __int64 v4; // [rsp+10h] [rbp-80h] BYREF
  __int64 v5; // [rsp+18h] [rbp-78h] BYREF
  _QWORD *v6; // [rsp+20h] [rbp-70h] BYREF
  char *v7; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v8[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v9[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 (__fastcall *v10)(const __m128i **, const __m128i *, int); // [rsp+50h] [rbp-40h]
  _QWORD *(__fastcall *v11)(_QWORD *, __int64 **, __int64); // [rsp+58h] [rbp-38h]
  _QWORD v12[6]; // [rsp+60h] [rbp-30h] BYREF

  v9[0] = a1 + 32;
  v9[1] = &v4;
  v4 = a1;
  v11 = sub_16D4420;
  v5 = a1;
  v10 = sub_16D4240;
  v7 = &v3;
  v3 = 0;
  v8[0] = sub_16D4680;
  v12[0] = v8;
  v12[1] = &v5;
  v12[2] = &v6;
  v6 = v9;
  v8[1] = 0;
  v12[3] = &v7;
  *(_QWORD *)(__readfsqword(0) - 24) = v12;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_16D42A0;
  if ( !&_pthread_key_create )
  {
    v1 = -1;
LABEL_11:
    sub_4264C5(v1);
  }
  v1 = pthread_once((pthread_once_t *)(a1 + 24), init_routine);
  if ( v1 )
    goto LABEL_11;
  if ( !v3 )
    sub_42641C(2u);
  if ( _InterlockedExchange((volatile __int32 *)(a1 + 16), 1) < 0 )
    sub_222D1B0();
  result = (__int64)v10;
  if ( v10 )
    return v10((const __m128i **)v9, (const __m128i *)v9, 3);
  return result;
}
