// Function: sub_6E2F40
// Address: 0x6e2f40
//
__int64 __fastcall sub_6E2F40(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int8 v2; // bl
  __int64 v3; // rax
  __int64 v5; // rax
  _QWORD *v6; // rax

  v1 = qword_4D03A80;
  v2 = a1;
  if ( qword_4D03A80 )
  {
    qword_4D03A80 = *(_QWORD *)qword_4D03A80;
  }
  else
  {
    a1 = 64;
    v1 = sub_823970(64);
  }
  *(_DWORD *)(v1 + 8) &= 0xFFFC00FF;
  *(_QWORD *)v1 = 0;
  *(_QWORD *)(v1 + 16) = 0;
  *(_BYTE *)(v1 + 8) = v2;
  if ( v2 == 2 )
  {
    *(_QWORD *)(v1 + 24) = 0;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    v5 = *(_QWORD *)&dword_4F077C8;
    *(_QWORD *)(v1 + 56) = 0;
    *(_QWORD *)(v1 + 48) = v5;
    return v1;
  }
  else if ( v2 > 2u )
  {
    if ( v2 != 3 )
      sub_721090(a1);
    *(_QWORD *)(v1 + 24) = 0;
    return v1;
  }
  else
  {
    if ( v2 )
    {
      *(_QWORD *)(v1 + 24) = 0;
      v3 = *(_QWORD *)&dword_4F077C8;
      *(_QWORD *)(v1 + 32) = *(_QWORD *)&dword_4F077C8;
      *(_QWORD *)(v1 + 40) = v3;
    }
    else
    {
      v6 = sub_6E2EF0();
      *(_QWORD *)(v1 + 32) = 0;
      *(_QWORD *)(v1 + 24) = v6;
    }
    return v1;
  }
}
