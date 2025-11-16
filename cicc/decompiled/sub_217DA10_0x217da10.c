// Function: sub_217DA10
// Address: 0x217da10
//
_BYTE *__fastcall sub_217DA10(__int64 a1, _BYTE *a2)
{
  _BYTE *v2; // rax
  _QWORD *v3; // r14
  _BYTE *v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // r12
  unsigned __int64 *v7; // rcx
  unsigned __int64 v8; // rdx

  if ( !a2 )
    BUG();
  v2 = a2;
  v3 = a2;
  if ( (*a2 & 4) == 0 && (a2[46] & 8) != 0 )
  {
    do
      v2 = (_BYTE *)*((_QWORD *)v2 + 1);
    while ( (v2[46] & 8) != 0 );
  }
  v4 = (_BYTE *)*((_QWORD *)v2 + 1);
  v5 = a1 + 16;
  if ( a2 != v4 )
  {
    do
    {
      v6 = v3;
      v3 = (_QWORD *)v3[1];
      sub_1DD5BC0(v5, (__int64)v6);
      v7 = (unsigned __int64 *)v6[1];
      v8 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
      *v7 = v8 | *v7 & 7;
      *(_QWORD *)(v8 + 8) = v7;
      *v6 &= 7uLL;
      v6[1] = 0;
      sub_1DD5C20(v5);
    }
    while ( v3 != (_QWORD *)v4 );
  }
  return v4;
}
