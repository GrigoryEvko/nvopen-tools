// Function: sub_2B17B70
// Address: 0x2b17b70
//
bool __fastcall sub_2B17B70(_DWORD **a1, _BYTE *a2, unsigned int *a3)
{
  unsigned int v3; // ebx
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned int v6; // r15d
  unsigned int v7; // r15d
  unsigned __int64 v8; // rax
  unsigned int v9; // edx
  unsigned int v10; // r8d
  bool result; // al
  unsigned __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-B8h]
  unsigned __int64 v15; // [rsp+18h] [rbp-A0h] BYREF
  unsigned int v16; // [rsp+20h] [rbp-98h]
  unsigned __int64 v17; // [rsp+28h] [rbp-90h] BYREF
  unsigned int v18; // [rsp+30h] [rbp-88h]
  __m128i v19; // [rsp+38h] [rbp-80h] BYREF
  __int64 v20; // [rsp+48h] [rbp-70h]
  __int64 v21; // [rsp+50h] [rbp-68h]
  __int64 v22; // [rsp+58h] [rbp-60h]
  __int64 v23; // [rsp+60h] [rbp-58h]
  __int64 v24; // [rsp+68h] [rbp-50h]
  __int64 v25; // [rsp+70h] [rbp-48h]
  __int16 v26; // [rsp+78h] [rbp-40h]

  v3 = *a1[2]
     - sub_9AF8B0((__int64)a2, *((_QWORD *)*a1 + 418), 0, *((_QWORD *)*a1 + 416), 0, *((_QWORD *)*a1 + 415), 1)
     - ((*(_BYTE *)a1[1] == 0)
      - 1);
  if ( *a2 <= 0x1Cu )
    goto LABEL_26;
  sub_D19730((__int64)&v15, *((_QWORD *)*a1 + 417), (__int64)a2, v13, v4, v5);
  v6 = v16;
  if ( v16 > 0x40 )
  {
    v7 = v6 - sub_C444A0((__int64)&v15);
    if ( !v7 )
      v7 = 1;
    if ( !*(_BYTE *)a1[1] )
      goto LABEL_17;
    if ( v3 > v7 )
      v3 = v7;
LABEL_24:
    if ( v15 )
      j_j___libc_free_0_0(v15);
    goto LABEL_26;
  }
  v7 = 1;
  if ( v15 )
  {
    _BitScanReverse64(&v12, v15);
    v7 = 64 - (v12 ^ 0x3F);
  }
  if ( *(_BYTE *)a1[1] )
  {
    if ( v3 > v7 )
      v3 = v7;
  }
  else
  {
LABEL_17:
    do
    {
      v9 = *a1[2];
      if ( v9 <= v7 )
        break;
      v18 = *a1[2];
      v10 = v7 - 1;
      if ( v9 <= 0x40 )
      {
        v17 = 0;
      }
      else
      {
        sub_C43690((__int64)&v17, 0, 0);
        v9 = v18;
        v10 = v7 - 1;
      }
      if ( v10 != v9 )
      {
        if ( v10 > 0x3F || v9 > 0x40 )
          sub_C43C90(&v17, v10, v9);
        else
          v17 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v7 + 63 - (unsigned __int8)v9) << v10;
      }
      v8 = *((_QWORD *)*a1 + 418);
      v20 = 0;
      v21 = 0;
      v19 = (__m128i)v8;
      v22 = 0;
      v23 = 0;
      v24 = 0;
      v25 = 0;
      v26 = 257;
      if ( (unsigned __int8)sub_9AC230((__int64)a2, (__int64)&v17, &v19, 0) )
      {
        if ( v18 > 0x40 && v17 )
          j_j___libc_free_0_0(v17);
        break;
      }
      v7 *= 2;
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
    }
    while ( !*(_BYTE *)a1[1] );
    if ( v3 > v7 )
      v3 = v7;
    if ( v16 > 0x40 )
      goto LABEL_24;
  }
LABEL_26:
  if ( *a3 >= v3 )
    v3 = *a3;
  *a3 = v3;
  result = 0;
  if ( v3 )
    return *a1[2] >= 2 * v3;
  return result;
}
