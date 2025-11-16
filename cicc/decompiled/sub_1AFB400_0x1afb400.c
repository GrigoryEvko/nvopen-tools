// Function: sub_1AFB400
// Address: 0x1afb400
//
__int64 __fastcall sub_1AFB400(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v15; // rcx
  __int64 v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // rax
  int v20; // ebx
  int v21; // ecx
  __int64 v22; // rsi
  _BYTE *v23; // r12
  _BYTE *v24; // r15
  unsigned __int64 v25; // r14
  unsigned int v26; // r12d
  __int64 v27; // r8
  _QWORD *v28; // rdi
  int v29; // eax
  __int64 v32; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+38h] [rbp-68h]
  _QWORD *v36; // [rsp+40h] [rbp-60h] BYREF
  __int64 v37; // [rsp+48h] [rbp-58h]
  _QWORD v38[10]; // [rsp+50h] [rbp-50h] BYREF

  v15 = 4;
  v17 = 1;
  v37 = 0x400000001LL;
  v36 = v38;
  v38[0] = a1;
  v18 = v38;
  v19 = 0;
  v20 = 0;
  v32 = a4;
  while ( 1 )
  {
    v22 = v18[v19];
    v23 = *(_BYTE **)(v22 + 16);
    v24 = *(_BYTE **)(v22 + 8);
    v25 = (v23 - v24) >> 3;
    if ( v25 > v15 - v17 )
    {
      sub_16CD150((__int64)&v36, v38, v25 + v17, 8, a5, a6);
      v18 = v36;
    }
    v21 = v37;
    if ( v24 != v23 )
    {
      memmove(&v18[(unsigned int)v37], v24, v23 - v24);
      v21 = v37;
      v18 = v36;
    }
    v19 = (unsigned int)(v20 + 1);
    LODWORD(v37) = v25 + v21;
    v17 = (unsigned int)(v25 + v21);
    v20 = v19;
    if ( (_DWORD)v19 == (_DWORD)v25 + v21 )
      break;
    v15 = HIDWORD(v37);
  }
  v26 = 0;
  if ( (_DWORD)v19 )
  {
    v27 = a5;
    do
    {
      v28 = (_QWORD *)v18[(unsigned int)v17 - 1];
      LODWORD(v37) = v17 - 1;
      v35 = v27;
      v29 = sub_1AF9650(v28, a2, a3, v32, v27, a6, a7, a8, a9, a10, a11, a12, a13, a14);
      LODWORD(v17) = v37;
      v18 = v36;
      v26 |= v29;
      v27 = v35;
    }
    while ( (_DWORD)v37 );
  }
  if ( v18 != v38 )
    _libc_free((unsigned __int64)v18);
  return v26;
}
