// Function: sub_2D283E0
// Address: 0x2d283e0
//
unsigned __int64 __fastcall sub_2D283E0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v4; // rdi
  __int64 v5; // rax

  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 == *(_QWORD *)(a1 + 40) + 48LL || !v1 )
    v2 = 0;
  else
    v2 = v1 - 24;
  if ( !sub_B44020(v2) )
    return v2 & 0xFFFFFFFFFFFFFFFBLL;
  v4 = *(_QWORD *)(v2 + 64);
  if ( v4 )
    v5 = sub_B14240(v4);
  else
    v5 = (__int64)&qword_4F81430[1];
  return v5 | 4;
}
