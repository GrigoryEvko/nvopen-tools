// Function: sub_1340D30
// Address: 0x1340d30
//
int __fastcall sub_1340D30(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v3; // rax

  if ( *((_BYTE *)a2 + 16) )
  {
    LODWORD(v3) = sub_1340AC0(a1, a2[1], a3);
  }
  else
  {
    a3[8] = a3;
    a3[9] = a3;
    v3 = *a2;
    if ( *a2 )
    {
      a3[8] = *(_QWORD *)(v3 + 72);
      *(_QWORD *)(*a2 + 72) = a3;
      a3[9] = *(_QWORD *)(a3[9] + 64LL);
      *(_QWORD *)(*(_QWORD *)(*a2 + 72) + 64LL) = *a2;
      v3 = a3[9];
      *(_QWORD *)(v3 + 64) = a3;
    }
    *a2 = (__int64)a3;
  }
  return v3;
}
