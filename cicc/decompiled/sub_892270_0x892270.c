// Function: sub_892270
// Address: 0x892270
//
__int64 __fastcall sub_892270(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 result; // rax

  v1 = a1[3];
  v3 = sub_87D510(v1, 0);
  if ( ((*(_BYTE *)(v1 + 80) - 7) & 0xFD) != 0 )
  {
    v5 = *(_QWORD *)(*(_QWORD *)sub_8CA280(v3) + 96LL);
    result = *(_QWORD *)(v5 + 16);
    if ( result )
      goto LABEL_3;
  }
  else
  {
    v4 = *(_QWORD *)sub_8C97D0(v3);
    v5 = sub_892240(v4);
    result = *(_QWORD *)(v5 + 16);
    if ( result )
    {
LABEL_3:
      a1[2] = result;
      return result;
    }
  }
  result = sub_880C20();
  *(_QWORD *)(result + 8) = a1;
  if ( a1[3] != a1[4] )
  {
    if ( qword_4F60200 )
      *(_QWORD *)qword_4F601F8 = result;
    else
      qword_4F60200 = result;
    qword_4F601F8 = result;
  }
  *(_QWORD *)(v5 + 16) = result;
  a1[2] = result;
  return result;
}
