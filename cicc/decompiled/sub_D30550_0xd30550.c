// Function: sub_D30550
// Address: 0xd30550
//
char __fastcall sub_D30550(
        unsigned __int8 *a1,
        __int64 a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  char result; // al
  char v9; // [rsp+Fh] [rbp-121h]
  __int64 v10; // [rsp+10h] [rbp-120h] BYREF
  char *v11; // [rsp+18h] [rbp-118h]
  __int64 v12; // [rsp+20h] [rbp-110h]
  int v13; // [rsp+28h] [rbp-108h]
  char v14; // [rsp+2Ch] [rbp-104h]
  char v15; // [rsp+30h] [rbp-100h] BYREF

  v10 = 0;
  v11 = &v15;
  v12 = 32;
  v13 = 0;
  v14 = 1;
  result = sub_D2F8A0(a1, a2, a3, a4, a5, a6, a7, a8, (__int64)&v10, 16);
  if ( !v14 )
  {
    v9 = result;
    _libc_free(v11, a2);
    return v9;
  }
  return result;
}
