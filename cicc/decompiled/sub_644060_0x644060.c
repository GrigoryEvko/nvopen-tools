// Function: sub_644060
// Address: 0x644060
//
void __fastcall sub_644060(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdx
  __int64 v3; // rax

  if ( *(char *)(a1 + 125) >= 0 && (*(_BYTE *)(a1 + 131) & 0x10) == 0 )
  {
    v1 = *(_QWORD **)(a1 + 368);
    if ( v1 )
    {
      do
      {
        v2 = v1;
        v1 = (_QWORD *)*v1;
      }
      while ( v1 );
      *v2 = qword_4CFDE68;
      v3 = *(_QWORD *)(a1 + 368);
      *(_QWORD *)(a1 + 368) = 0;
      qword_4CFDE68 = v3;
    }
  }
}
