// Function: sub_71B3F0
// Address: 0x71b3f0
//
_QWORD *__fastcall sub_71B3F0(__int64 a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // rax
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax

  v1 = (_QWORD *)sub_726B30(8);
  *v1 = *(_QWORD *)dword_4F07508;
  v2 = sub_6EFF80();
  v1[6] = sub_73A7B0(v2);
  v3 = (_QWORD *)sub_726B30(1);
  *v3 = *(_QWORD *)dword_4F07508;
  v4 = sub_6EFF80();
  v5 = sub_73DBF0(29, v4, a1);
  v3[9] = v1;
  v3[6] = v5;
  v1[3] = v3;
  return v3;
}
