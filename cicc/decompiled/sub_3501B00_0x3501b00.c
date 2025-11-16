// Function: sub_3501B00
// Address: 0x3501b00
//
__int64 __fastcall sub_3501B00(unsigned int *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int16 v5; // si
  __int64 result; // rax
  __int64 v7; // rcx
  unsigned int v8; // esi
  __int64 i; // rdx

  v3 = *a1;
  ++a1[1];
  *((_QWORD *)a1 + 5) = 0;
  LODWORD(v3) = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 24 * v3 + 16);
  v5 = v3;
  result = (unsigned int)v3 >> 12;
  v7 = *(_QWORD *)(a3 + 56) + 2 * result;
  v8 = v5 & 0xFFF;
  for ( i = 0; v7; i = (unsigned int)(i + 1) )
  {
    v7 += 2;
    result = *((_QWORD *)a1 + 6) + 112 * i;
    *(_DWORD *)(result + 88) = *(_DWORD *)(a2 + 216LL * v8);
    v8 += *(__int16 *)(v7 - 2);
    if ( !*(_WORD *)(v7 - 2) )
      break;
  }
  return result;
}
