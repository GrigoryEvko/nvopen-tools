// Function: sub_2712480
// Address: 0x2712480
//
unsigned __int64 *__fastcall sub_2712480(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v6; // r12
  __int16 v7; // dx
  unsigned __int64 *result; // rax

  v6 = a1[1];
  if ( v6 == a1[2] )
    return sub_2712140(a1, a1[1], (__int64 *)a2, a4, a5, a6);
  if ( v6 )
  {
    *(_QWORD *)v6 = *(_QWORD *)a2;
    v7 = *(_WORD *)(a2 + 8);
    *(_BYTE *)(v6 + 10) = *(_BYTE *)(a2 + 10);
    *(_WORD *)(v6 + 8) = v7;
    *(_WORD *)(v6 + 16) = *(_WORD *)(a2 + 16);
    *(_QWORD *)(v6 + 24) = *(_QWORD *)(a2 + 24);
    sub_C8CF70(v6 + 32, (void *)(v6 + 64), 2, a2 + 64, a2 + 32);
    sub_C8CF70(v6 + 80, (void *)(v6 + 112), 2, a2 + 112, a2 + 80);
    result = (unsigned __int64 *)*(unsigned __int8 *)(a2 + 128);
    *(_BYTE *)(v6 + 128) = (_BYTE)result;
    v6 = a1[1];
  }
  a1[1] = v6 + 136;
  return result;
}
