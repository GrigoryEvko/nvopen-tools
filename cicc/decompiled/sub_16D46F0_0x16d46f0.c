// Function: sub_16D46F0
// Address: 0x16d46f0
//
__int64 __fastcall sub_16D46F0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // rax
  _QWORD *v5; // r12
  pthread_once_t *v6; // rdi
  unsigned int v7; // eax
  volatile signed __int32 *v8; // rdi
  __int64 result; // rax
  char v10; // [rsp+Fh] [rbp-A1h] BYREF
  __int64 v11; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-98h] BYREF
  _QWORD *v13; // [rsp+20h] [rbp-90h] BYREF
  char *v14; // [rsp+28h] [rbp-88h] BYREF
  _QWORD v15[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v16[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 (__fastcall *v17)(const __m128i **, const __m128i *, int); // [rsp+50h] [rbp-60h]
  _QWORD *(__fastcall *v18)(_QWORD *, __int64 **, __int64); // [rsp+58h] [rbp-58h]
  _QWORD v19[10]; // [rsp+60h] [rbp-50h] BYREF

  v16[0] = a1 + 32;
  v2 = *a2;
  v16[1] = &v11;
  v3 = a2[1];
  *a2 = 0;
  a2[1] = 0;
  v18 = sub_16D4460;
  v11 = a1;
  v17 = sub_16D4270;
  v10 = 0;
  v4 = sub_22077B0(32);
  v5 = (_QWORD *)v4;
  if ( v4 )
  {
    *(_QWORD *)(v4 + 16) = 0;
    *(_QWORD *)(v4 + 24) = 0;
  }
  v12 = a1;
  v14 = &v10;
  v13 = v16;
  v15[0] = sub_16D4680;
  v19[0] = v15;
  v19[1] = &v12;
  v19[2] = &v13;
  v15[1] = 0;
  v19[3] = &v14;
  *(_QWORD *)(__readfsqword(0) - 24) = v19;
  v6 = (pthread_once_t *)(a1 + 24);
  *(_QWORD *)(__readfsqword(0) - 32) = sub_16D42A0;
  if ( !&_pthread_key_create )
  {
    v7 = -1;
LABEL_13:
    sub_4264C5(v7);
  }
  v7 = pthread_once(v6, init_routine);
  if ( v7 )
    goto LABEL_13;
  if ( !v10 )
    sub_42641C(2u);
  v8 = (volatile signed __int32 *)v5[3];
  v5[2] = v2;
  if ( v8 && !_InterlockedSub(v8 + 3, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v8 + 24LL))(v8);
  v5[3] = v3;
  sub_222D620(v5);
  result = (__int64)v17;
  if ( v17 )
    return v17((const __m128i **)v16, (const __m128i *)v16, 3);
  return result;
}
