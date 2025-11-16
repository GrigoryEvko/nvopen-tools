// Function: sub_7F7F20
// Address: 0x7f7f20
//
__int64 sub_7F7F20()
{
  __int64 result; // rax
  __int64 v1; // r13
  __int64 v2; // rbx
  _QWORD *v3; // rax
  _QWORD *v4; // r12

  result = qword_4F18B10;
  if ( !qword_4F18B10 )
  {
    v1 = sub_7E1C10();
    v2 = sub_72CBE0();
    v3 = sub_7259C0(7);
    v3[20] = v2;
    v4 = v3;
    *(_BYTE *)(v3[21] + 16LL) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v3[21] + 16LL) & 0xFD;
    if ( v1 )
      *(_QWORD *)v3[21] = sub_724EF0(v1);
    result = sub_72D2E0(v4);
    qword_4F18B10 = result;
  }
  return result;
}
