// Function: sub_130B510
// Address: 0x130b510
//
__int64 __fastcall sub_130B510(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned int a6,
        unsigned __int8 a7,
        unsigned __int8 a8,
        __int64 a9)
{
  __int64 v9; // r10
  __int64 v12; // r8
  __int64 result; // rax
  unsigned int v14; // r15d
  __int64 v15; // rcx
  unsigned int v16; // [rsp+8h] [rbp-48h]
  int v17; // [rsp+10h] [rbp-40h]
  __int64 v18; // [rsp+10h] [rbp-40h]
  _QWORD *v20; // [rsp+18h] [rbp-38h]

  v9 = a5;
  v12 = a7;
  if ( a8
    || !*(_BYTE *)(a2 + 16)
    || (v16 = v9,
        v18 = a4,
        result = (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, __int64, _QWORD, _QWORD, __int64, __int64))(a2 + 62264))(
                   a1,
                   a2 + 62264,
                   a3,
                   a4,
                   a7,
                   0,
                   v9,
                   a9),
        v12 = a7,
        v9 = v16,
        a4 = v18,
        !result) )
  {
    v17 = v9;
    result = (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, __int64, __int64, _QWORD, __int64, __int64))(a2 + 24))(
               a1,
               a2 + 24,
               a3,
               a4,
               v12,
               a8,
               v9,
               a9);
    if ( !result )
      return result;
    LODWORD(v9) = v17;
  }
  _InterlockedAdd64((volatile signed __int64 *)(a2 + 8), a3 >> 12);
  v14 = a6;
  v15 = a6;
  v20 = (_QWORD *)result;
  sub_1342130(a1, *(_QWORD *)(a2 + 68264), result, v15, (unsigned int)v9);
  result = (__int64)v20;
  *v20 = ((unsigned __int64)a5 << 12) | ((unsigned __int64)v14 << 20) | *v20 & 0xFFFFFFFFF00FEFFFLL;
  if ( a3 > 0x2000 )
  {
    if ( a5 )
    {
      sub_1341C60(a1, *(_QWORD *)(a2 + 68264), v20, v14);
      return (__int64)v20;
    }
  }
  return result;
}
