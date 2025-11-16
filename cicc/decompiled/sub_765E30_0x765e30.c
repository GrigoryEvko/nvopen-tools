// Function: sub_765E30
// Address: 0x765e30
//
void __fastcall sub_765E30(_QWORD *a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // r12
  char v3; // al

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1[1];
      v3 = *(_BYTE *)(v2 + 178);
      if ( (v3 & 0x40) != 0
        || v3 < 0
        || (*(_BYTE *)(v2 + 89) & 1) != 0
        || *(char *)(v2 + 141) < 0
        || *(_QWORD *)(v2 + 96) )
      {
        sub_760BD0((_QWORD *)v1[1], 6);
        sub_75BF90(v2);
      }
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
  }
}
