// Function: sub_2232A70
// Address: 0x2232a70
//
__int64 __fastcall sub_2232A70(__int64 a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rbp

  v2 = sub_22091A0(&qword_4FD69B8);
  v3 = *(_QWORD *)(*a2 + 24) + 8 * v2;
  result = *(_QWORD *)v3;
  if ( !*(_QWORD *)v3 )
  {
    v5 = sub_22077B0(0x90u);
    *(_DWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = 0;
    *(_QWORD *)v5 = off_4A04910;
    *(_BYTE *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 64) = 0;
    *(_WORD *)(v5 + 72) = 0;
    *(_BYTE *)(v5 + 136) = 0;
    sub_222FDB0(v5, a2);
    sub_2209690(*a2, (volatile signed __int32 *)v5, v2);
    return *(_QWORD *)v3;
  }
  return result;
}
