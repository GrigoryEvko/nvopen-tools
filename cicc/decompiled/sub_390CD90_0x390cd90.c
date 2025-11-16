// Function: sub_390CD90
// Address: 0x390cd90
//
void (*__fastcall sub_390CD90(__int64 a1, __int64 a2))()
{
  __int64 v3; // r12
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // rdi
  void (*result)(); // rax

  v3 = *(unsigned int *)(a2 + 16);
  if ( (_DWORD)v3 )
  {
    v4 = 8 * v3;
    v5 = 0;
    do
    {
      v6 = *(_QWORD *)(*(_QWORD *)(a2 + 8) + v5);
      v5 += 8;
      sub_38D01B0(a2, *(_QWORD *)(v6 + 96) & 0xFFFFFFFFFFFFFFF8LL);
      sub_390B580(a1, (_QWORD *)a2, *(_QWORD *)(v6 + 96) & 0xFFFFFFFFFFFFFFF8LL);
    }
    while ( v4 != v5 );
  }
  v7 = *(_QWORD *)(a1 + 8);
  result = *(void (**)())(*(_QWORD *)v7 + 128LL);
  if ( result != nullsub_1966 )
    return (void (*)())((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v7, a1, a2);
  return result;
}
