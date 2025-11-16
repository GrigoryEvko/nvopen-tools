// Function: sub_67D0F0
// Address: 0x67d0f0
//
__int64 __fastcall sub_67D0F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx

  v1 = sub_823970(160);
  *(_QWORD *)v1 = a1;
  *(_WORD *)(v1 + 24) = 0;
  v2 = qword_4CFFD90;
  *(_QWORD *)(v1 + 152) = 100;
  *(_QWORD *)(v1 + 8) = 0;
  *(_QWORD *)(v1 + 16) = v2;
  if ( v2 )
    *(_QWORD *)(v2 + 8) = v1;
  else
    qword_4CFFD88 = v1;
  qword_4CFFD90 = v1;
  return 100;
}
