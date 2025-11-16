// Function: sub_8C9930
// Address: 0x8c9930
//
__int64 __fastcall sub_8C9930(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  __int64 v4; // rdi
  _QWORD *v5; // r12
  __int64 v6; // rdi
  char v7; // al
  __int64 result; // rax

  v2 = a1;
  v3 = sub_878440();
  v4 = *(_QWORD *)(a1 + 104);
  v5 = v3;
  if ( v4 )
  {
    v6 = *(_QWORD *)sub_8C9880(v4);
    v7 = *(_BYTE *)(a2 + 80);
    if ( v7 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v7 - 4) <= 2u )
      v6 = sub_892920(v6);
    v2 = *(_QWORD *)(v6 + 88);
  }
  result = *(_QWORD *)(v2 + 112);
  *v5 = result;
  *(_QWORD *)(v2 + 112) = v5;
  v5[1] = a2;
  return result;
}
