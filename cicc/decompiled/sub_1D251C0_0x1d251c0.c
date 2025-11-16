// Function: sub_1D251C0
// Address: 0x1d251c0
//
__int64 __fastcall sub_1D251C0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int128 a11,
        __int64 a12,
        unsigned int a13,
        unsigned int a14,
        __int64 a15)
{
  __int64 v16; // rcx
  __int64 v19; // r8
  unsigned int v20; // ebx
  int v21; // eax
  int v22; // ebx
  __int64 v23; // rax
  unsigned int v25; // eax
  __int16 v26; // [rsp+4h] [rbp-3Ch]
  int v27; // [rsp+8h] [rbp-38h]
  unsigned int v28; // [rsp+8h] [rbp-38h]
  unsigned __int16 v29; // [rsp+Ch] [rbp-34h]

  v16 = (unsigned int)a6;
  v19 = a13;
  v20 = a14;
  v29 = a2;
  if ( !(_DWORD)a6 )
  {
    a2 = (unsigned int)a9;
    v28 = a13;
    v25 = sub_1D172F0((__int64)a1, (unsigned int)a9, a10);
    v19 = v28;
    v16 = v25;
  }
  if ( !v20 )
  {
    if ( (_BYTE)a9 )
    {
      v22 = sub_1D13440(a9);
    }
    else
    {
      v26 = v19;
      v27 = v16;
      v21 = sub_1F58D40(&a9, a2, a3, v16, v19, a6);
      LOWORD(v19) = v26;
      LODWORD(v16) = v27;
      v22 = v21;
    }
    v20 = (unsigned int)(v22 + 7) >> 3;
  }
  v23 = sub_1E0B8E0(a1[4], (unsigned __int16)v19, v20, v16, a15, 0, a11, a12, 1, 0, 0);
  return sub_1D24DC0(a1, v29, a3, a4, a5, v23, a7, a8, a9, a10);
}
