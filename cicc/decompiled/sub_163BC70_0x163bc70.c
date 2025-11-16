// Function: sub_163BC70
// Address: 0x163bc70
//
__int64 __fastcall sub_163BC70(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 result; // rax
  int v6; // ebx
  int v7; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4F9EE98, 1, 0) )
  {
    do
    {
      v6 = dword_4F9EE98;
      result = sub_16AF4B0();
      if ( v6 == 2 )
        break;
      v7 = dword_4F9EE98;
      result = sub_16AF4B0();
    }
    while ( v7 != 2 );
  }
  else
  {
    sub_15CD350(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 21;
      *(_QWORD *)v1 = "Safepoint IR Verifier";
      *(_QWORD *)(v1 + 16) = "verify-safepoint-ir";
      *(_QWORD *)(v1 + 32) = &unk_4F9EE9C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 19;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_163BD60;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    result = sub_16AF4B0();
    dword_4F9EE98 = 2;
  }
  return result;
}
