// Function: sub_39F13D0
// Address: 0x39f13d0
//
void __fastcall sub_39F13D0(__int64 a1, int *a2, __int64 a3)
{
  unsigned __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi

  sub_38D47A0(a1, a2, a3);
  v3 = sub_38D4B30(a1);
  v4 = *(unsigned int *)(v3 + 96);
  if ( (_DWORD)v4 )
  {
    v5 = 0;
    v6 = 24 * v4;
    do
    {
      v7 = *(_QWORD *)(*(_QWORD *)(v3 + 88) + v5);
      v5 += 24;
      sub_39EFF60(a1, v7);
    }
    while ( v6 != v5 );
  }
}
