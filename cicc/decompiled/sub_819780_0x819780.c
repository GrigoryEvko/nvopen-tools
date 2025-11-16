// Function: sub_819780
// Address: 0x819780
//
__int64 sub_819780()
{
  __int64 v0; // r12
  __int64 v2; // rax
  __int64 v3; // rax

  v0 = qword_4F194E8;
  if ( qword_4F194E8 )
  {
    qword_4F194E8 = *(_QWORD *)qword_4F194E8;
  }
  else
  {
    v2 = sub_822B10(88);
    *(_QWORD *)(v2 + 24) = 400;
    v0 = v2;
    v3 = sub_822BE0(401);
    *(_QWORD *)(v0 + 72) = 800;
    *(_QWORD *)(v0 + 16) = v3;
    *(_QWORD *)(v0 + 64) = sub_822BE0(801);
  }
  *(_QWORD *)v0 = 0;
  *(_WORD *)(v0 + 80) = 0;
  *(_QWORD *)(v0 + 8) = 0;
  *(_QWORD *)(v0 + 32) = 0;
  *(_QWORD *)(v0 + 40) = 0;
  *(_QWORD *)(v0 + 48) = 0;
  *(_QWORD *)(v0 + 56) = 0;
  return v0;
}
