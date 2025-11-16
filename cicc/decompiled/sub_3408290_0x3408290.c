// Function: sub_3408290
// Address: 0x3408290
//
__int64 __fastcall sub_3408290(
        __int64 a1,
        _QWORD *a2,
        __int128 *a3,
        __int64 a4,
        unsigned int *a5,
        unsigned int *a6,
        __m128i a7)
{
  __int128 v11; // rax
  __int64 v12; // r9
  int v13; // eax
  int v14; // edx
  __int64 v15; // rsi
  __int128 v16; // rax
  __int64 v17; // r9
  int v19; // edx
  int v20; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v21; // [rsp+10h] [rbp-50h]

  *(_QWORD *)&v11 = sub_3400EE0((__int64)a2, 0, a4, 0, a7);
  v21 = sub_3406EB0(a2, 0xA1u, a4, *a5, *((_QWORD *)a5 + 1), v12, *a3, v11);
  v13 = *(unsigned __int16 *)a5;
  v20 = v14;
  if ( (_WORD)v13 )
    v15 = word_4456340[v13 - 1];
  else
    v15 = (unsigned int)sub_3007240((__int64)a5);
  *(_QWORD *)&v16 = sub_3400EE0((__int64)a2, v15, a4, 0, a7);
  *(_QWORD *)(a1 + 16) = sub_3406EB0(a2, 0xA1u, a4, *a6, *((_QWORD *)a6 + 1), v17, *a3, v16);
  *(_QWORD *)a1 = v21;
  *(_DWORD *)(a1 + 24) = v19;
  *(_DWORD *)(a1 + 8) = v20;
  return a1;
}
