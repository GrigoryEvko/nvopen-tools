// Function: sub_690B80
// Address: 0x690b80
//
__int64 __fastcall sub_690B80(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v3; // rax
  __int64 v4; // rdi
  _DWORD *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rbx
  char v11; // r12

  result = unk_4D03C50;
  if ( !unk_4D03C50 )
    goto LABEL_8;
  if ( (*(_BYTE *)(unk_4D03C50 + 21LL) & 8) != 0 )
    return result;
  v3 = *(__int64 **)(unk_4D03C50 + 136LL);
  if ( v3 && (v4 = *v3) != 0 )
  {
    v5 = (_DWORD *)sub_6E1A20(v4);
    if ( (unsigned int)sub_6E5430(v4, a2, v6, v7, v8, v9) )
      sub_6851C0(0x12u, v5);
    return sub_6E1BF0(*(_QWORD *)(unk_4D03C50 + 136LL));
  }
  else
  {
LABEL_8:
    v10 = qword_4F061C8;
    v11 = *(_BYTE *)(qword_4F061C8 + 75LL);
    *(_BYTE *)(qword_4F061C8 + 75LL) = 0;
    result = sub_7BE280(28, 18, 0, 0);
    *(_BYTE *)(v10 + 75) = v11;
  }
  return result;
}
