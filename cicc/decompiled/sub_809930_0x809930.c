// Function: sub_809930
// Address: 0x809930
//
unsigned __int8 *__fastcall sub_809930(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  char v4; // r13
  __int64 v5; // r15
  __int64 v6; // rax

  if ( unk_4D042D0 && unk_4D042D0 < (unsigned __int64)(*(_QWORD *)a3 - 1LL) )
  {
    v4 = *(_BYTE *)(a3 + 60);
    v5 = unk_4D042D0 - 10LL;
    if ( !v4 )
      v4 = 95;
    v6 = sub_723DE0(a1, 0);
    sprintf((char *)&a1[v5], "_%c%08lx", (unsigned int)v4, v6);
    *(_QWORD *)a3 = unk_4D042D0 + 1LL;
    if ( a2 )
      *(_BYTE *)(a2 + 89) |= 0x10u;
  }
  return a1;
}
