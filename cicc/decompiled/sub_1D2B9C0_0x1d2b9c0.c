// Function: sub_1D2B9C0
// Address: 0x1d2b9c0
//
__int64 *__fastcall sub_1D2B9C0(
        _QWORD *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int64 a10,
        int a11,
        int a12,
        unsigned __int8 a13)
{
  unsigned __int8 v16; // cl
  __int64 v17; // rbx
  __int64 v18; // rsi
  int v19; // edx
  int v20; // ecx
  int v21; // eax
  unsigned int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rax
  int v26; // eax
  int v27; // eax
  int v28; // [rsp+Ch] [rbp-84h]
  __int64 v29; // [rsp+10h] [rbp-80h] BYREF
  __int64 v30; // [rsp+18h] [rbp-78h]
  __int128 v31; // [rsp+20h] [rbp-70h]
  __int64 v32; // [rsp+30h] [rbp-60h]
  _QWORD v33[3]; // [rsp+40h] [rbp-50h] BYREF

  v29 = a4;
  v30 = a5;
  v16 = a13;
  if ( !a11 )
  {
    v27 = sub_1D172F0((__int64)a1, (unsigned int)v29, a5);
    v16 = a13;
    a11 = v27;
  }
  v17 = a1[4];
  v18 = 6;
  if ( a2 != 220 )
    v18 = 2 * (unsigned int)(a2 != 219) + 5;
  memset(v33, 0, sizeof(v33));
  if ( (_BYTE)v29 )
  {
    v19 = sub_1D13440(v29);
  }
  else
  {
    v28 = v16;
    v26 = sub_1F58D40(&v29, v18, a3, v16, a5, a6);
    v20 = v28;
    v19 = v26;
  }
  v31 = (unsigned __int64)a6;
  v21 = 0;
  v22 = (unsigned int)(v19 + 7) >> 3;
  LOBYTE(v32) = 0;
  if ( a6 )
  {
    v23 = *a6;
    if ( *(_BYTE *)(*a6 + 8) == 16 )
      v23 = **(_QWORD **)(v23 + 16);
    v21 = *(_DWORD *)(v23 + 8) >> 8;
  }
  HIDWORD(v32) = v21;
  v24 = sub_1E0B8E0(v17, v18, v22, a11, (unsigned int)v33, 0, v31, v32, v20, a12, 0);
  return sub_1D2B8F0(a1, a2, a3, (unsigned int)v29, v30, v24, a7, a8, a9, a10);
}
