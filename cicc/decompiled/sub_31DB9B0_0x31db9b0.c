// Function: sub_31DB9B0
// Address: 0x31db9b0
//
void __fastcall sub_31DB9B0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax

  if ( *(_QWORD *)(a1 + 768) )
  {
    v2 = *(_QWORD **)(a2 + 32);
    v3 = v2[3];
    v4 = v2[8];
    v5 = v2[13];
    v6 = v2[18];
    v7 = sub_B10CD0(a2 + 56);
    sub_3256960(*(_QWORD *)(a1 + 768), v3, v4, v5, v6, v7);
  }
}
