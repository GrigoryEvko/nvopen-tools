// Function: sub_770440
// Address: 0x770440
//
_QWORD *__fastcall sub_770440(__int64 a1)
{
  int v1; // esi
  int v2; // eax
  __int64 v3; // rcx
  __int64 v4; // rdx

  v1 = *(_DWORD *)(a1 + 8);
  if ( !v1 )
  {
    v3 = 0;
    goto LABEL_6;
  }
  v2 = *(_DWORD *)(a1 + 8);
  LODWORD(v3) = 0;
  do
  {
    v4 = (unsigned int)(v2 - 1);
    v3 = (unsigned int)(v3 + 1);
    v2 &= v4;
  }
  while ( v2 );
  if ( (int)v3 <= 10 )
  {
    v3 = (int)v3;
LABEL_6:
    **(_QWORD **)a1 = qword_4F08320[v3];
    qword_4F08320[v3] = *(_QWORD *)a1;
    return qword_4F08320;
  }
  return (_QWORD *)sub_822B90(*(_QWORD *)a1, (unsigned int)(16 * (v1 + 1)), v4, v3);
}
