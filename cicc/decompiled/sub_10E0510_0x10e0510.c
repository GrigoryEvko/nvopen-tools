// Function: sub_10E0510
// Address: 0x10e0510
//
__int64 __fastcall sub_10E0510(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v13; // rdi
  __int64 v14; // rcx
  int v15; // esi
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r14
  int v22; // [rsp+2Ch] [rbp-34h]

  v13 = a5 + 56 * a6;
  if ( a5 == v13 )
  {
    v15 = 0;
  }
  else
  {
    v14 = a5;
    v15 = 0;
    do
    {
      v16 = *(_QWORD *)(v14 + 40) - *(_QWORD *)(v14 + 32);
      v14 += 56;
      v15 += v16 >> 3;
    }
    while ( v13 != v14 );
  }
  LOBYTE(v22) = 16 * (_DWORD)a6 != 0;
  v17 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * a6) << 32) | (unsigned int)(v15 + a4 + 1));
  v18 = (__int64)v17;
  if ( v17 )
  {
    sub_B44260((__int64)v17, **(_QWORD **)(a1 + 16), 56, (v22 << 28) | (v15 + a4 + 1) & 0x7FFFFFF, a8, a9);
    *(_QWORD *)(v18 + 72) = 0;
    sub_B4A290(v18, a1, a2, a3, a4, a7, a5, a6);
  }
  return v18;
}
