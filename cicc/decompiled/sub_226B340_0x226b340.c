// Function: sub_226B340
// Address: 0x226b340
//
_QWORD *__fastcall sub_226B340(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *result; // rax
  _QWORD *v4; // rdi

  *(_QWORD *)a1 = a2;
  v2 = (_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 1;
  do
  {
    if ( v2 )
    {
      *v2 = -4;
      v2[1] = -3;
      v2[2] = -4;
      v2[3] = -3;
    }
    v2 += 5;
  }
  while ( (_QWORD *)(a1 + 352) != v2 );
  *(_QWORD *)(a1 + 352) = a1 + 528;
  *(_QWORD *)(a1 + 376) = a1 + 392;
  *(_QWORD *)(a1 + 384) = 0x400000000LL;
  *(_WORD *)(a1 + 520) = 256;
  *(_QWORD *)(a1 + 360) = 0;
  *(_BYTE *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 528) = &unk_49DDBE8;
  result = (_QWORD *)(a1 + 552);
  v4 = (_QWORD *)(a1 + 680);
  *(v4 - 18) = 0;
  *(v4 - 17) = 1;
  do
  {
    if ( result )
      *result = -4096;
    result += 2;
  }
  while ( result != v4 );
  return result;
}
