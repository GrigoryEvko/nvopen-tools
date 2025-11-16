// Function: sub_1169330
// Address: 0x1169330
//
unsigned __int8 *__fastcall sub_1169330(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 *v3; // r12
  __int64 v4; // r13
  unsigned int v6; // r14d
  __int64 v7; // rbx

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v2 = *(_QWORD *)(a2 - 8);
  else
    v2 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v3 = *(unsigned __int8 **)v2;
  v4 = *(_QWORD *)(v2 + 32);
  if ( sub_B46D50((unsigned __int8 *)a2)
    && ((v6 = sub_F13260(v3), (*(_BYTE *)(a2 + 7) & 0x40) != 0)
      ? (v7 = *(_QWORD *)(a2 - 8))
      : (v7 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
        v6 < (unsigned int)sub_F13260(*(unsigned __int8 **)(v7 + 32))) )
  {
    return (unsigned __int8 *)v4;
  }
  else
  {
    return v3;
  }
}
