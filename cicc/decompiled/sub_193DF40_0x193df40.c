// Function: sub_193DF40
// Address: 0x193df40
//
__int64 __fastcall sub_193DF40(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  int v4; // eax
  unsigned int v5; // r8d
  unsigned __int8 v7; // [rsp+7h] [rbp-29h] BYREF
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = 0;
  v3 = a1 + 8;
  if ( *(void **)(a1 + 8) == sub_16982C0() )
    v4 = sub_169E030(v3, v8, 1, 0x40u, 1u, 3u, &v7);
  else
    v4 = sub_169A0A0(v3, v8, 1, 0x40u, 1u, 3u, &v7);
  v5 = 0;
  if ( !v4 )
  {
    v5 = v7;
    if ( v7 )
      *a2 = v8[0];
  }
  return v5;
}
