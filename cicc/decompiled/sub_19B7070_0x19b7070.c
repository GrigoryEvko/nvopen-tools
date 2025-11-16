// Function: sub_19B7070
// Address: 0x19b7070
//
__int64 __fastcall sub_19B7070(__int64 a1, _DWORD *a2, _BYTE *a3, _BYTE *a4, __int64 *a5, __int64 a6, int a7)
{
  __int64 *v7; // rax
  __int64 *v8; // r13
  __int64 *v11; // r15
  __int64 v12; // rsi
  int v13; // edx
  char v14; // si
  char v15; // cl
  __int64 v16; // rdi
  unsigned int v17; // r12d
  int v22; // [rsp+20h] [rbp-80h] BYREF
  char v23; // [rsp+24h] [rbp-7Ch]
  __int64 v24; // [rsp+28h] [rbp-78h]
  __int64 v25; // [rsp+30h] [rbp-70h]
  __int64 v26; // [rsp+38h] [rbp-68h]
  __int64 v27; // [rsp+40h] [rbp-60h]
  int v28; // [rsp+48h] [rbp-58h]
  __int64 v29; // [rsp+50h] [rbp-50h]
  __int64 v30; // [rsp+58h] [rbp-48h]
  __int64 v31; // [rsp+60h] [rbp-40h]
  int v32; // [rsp+68h] [rbp-38h]

  v7 = *(__int64 **)(a1 + 32);
  v8 = *(__int64 **)(a1 + 40);
  v23 = 0;
  v22 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  if ( v8 == v7 )
  {
    v16 = 0;
    v17 = 0;
    v15 = 0;
    v14 = 0;
    v13 = 0;
  }
  else
  {
    v11 = v7;
    do
    {
      v12 = *v11++;
      sub_14D0990((__int64)&v22, v12, a5, a6);
    }
    while ( v8 != v11 );
    v13 = HIDWORD(v30);
    v14 = BYTE2(v22);
    v15 = HIBYTE(v22);
    v16 = v26;
    v17 = v24 - v32;
  }
  *a2 = v13;
  if ( byte_4FB24C0 )
    *a2 = v13 + HIDWORD(v29) + v30;
  *a3 = v14;
  *a4 = v15;
  if ( a7 + 1 >= v17 )
    v17 = a7 + 1;
  j___libc_free_0(v16);
  return v17;
}
