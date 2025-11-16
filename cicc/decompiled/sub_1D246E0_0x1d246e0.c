// Function: sub_1D246E0
// Address: 0x1d246e0
//
__int64 *__fastcall sub_1D246E0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11,
        __int128 a12,
        __int128 a13,
        __int64 a14,
        int a15,
        int a16,
        unsigned __int8 a17)
{
  unsigned __int16 v18; // r13
  int v19; // ebx
  unsigned __int8 v20; // cl
  __int64 v21; // r15
  int v22; // edx
  int v23; // ecx
  __int64 v24; // rax
  int v26; // eax
  int v27; // eax
  int v28; // [rsp+Ch] [rbp-64h]
  __int64 v29; // [rsp+10h] [rbp-60h] BYREF
  __int64 v30; // [rsp+18h] [rbp-58h]
  _QWORD v31[3]; // [rsp+20h] [rbp-50h] BYREF

  v18 = a2;
  v19 = a6;
  v29 = a4;
  v20 = a17;
  v30 = a5;
  if ( !(_DWORD)a6 )
  {
    a2 = (unsigned int)v29;
    v27 = sub_1D172F0((__int64)a1, (unsigned int)v29, a5);
    v20 = a17;
    v19 = v27;
  }
  v21 = a1[4];
  memset(v31, 0, sizeof(v31));
  if ( (_BYTE)v29 )
  {
    v22 = sub_1D13440(v29);
  }
  else
  {
    v28 = v20;
    v26 = sub_1F58D40(&v29, a2, a3, v20, a5, a6);
    v23 = v28;
    v22 = v26;
  }
  v24 = sub_1E0B8E0(v21, 7, (unsigned int)(v22 + 7) >> 3, v19, (unsigned int)v31, 0, a13, a14, v23, a15, a16);
  return sub_1D24690(a1, v18, a3, v29, v30, v24, a7, a8, a9, a10, a11, a12);
}
