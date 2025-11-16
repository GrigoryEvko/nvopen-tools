// Function: sub_222C4A0
// Address: 0x222c4a0
//
__int64 __fastcall sub_222C4A0(__int64 a1, __int64 a2, __int64 a3)
{
  bool v5; // zf
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax

  if ( !sub_2207CD0((_QWORD *)(a1 + 104)) )
    return -1;
  if ( *(_BYTE *)(a1 + 192) )
  {
    v5 = *(_QWORD *)(a1 + 16) == *(_QWORD *)(a1 + 8);
    *(_BYTE *)(a1 + 192) = 0;
    v6 = *(_QWORD *)(a1 + 184);
    v7 = *(_QWORD *)(a1 + 152);
    v8 = *(_QWORD *)(a1 + 176) + !v5;
    *(_QWORD *)(a1 + 176) = v8;
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = v8;
    *(_QWORD *)(a1 + 24) = v6;
  }
  return sub_222BFB0(a1, a2, 0, a3);
}
