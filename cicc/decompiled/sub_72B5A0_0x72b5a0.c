// Function: sub_72B5A0
// Address: 0x72b5a0
//
_QWORD *__fastcall sub_72B5A0(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v4; // r8
  __int64 i; // rax
  __int64 v6; // rbx

  v4 = sub_7259C0(15);
  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v6 = *(_QWORD *)(i + 128) * a2;
  v4[20] = a1;
  *((_BYTE *)v4 + 177) = a3;
  v4[16] = v6;
  *((_DWORD *)v4 + 34) = v6;
  return v4;
}
