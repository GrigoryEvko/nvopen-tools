// Function: sub_6F8E70
// Address: 0x6f8e70
//
__int64 __fastcall sub_6F8E70(__int64 a1, _QWORD *a2, _QWORD *a3, _QWORD *a4, __int64 a5)
{
  __int64 v7; // r15
  __int64 *v8; // rax
  bool v9; // zf
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9

  v7 = *(_QWORD *)(a1 + 120);
  if ( (dword_4D04964 || dword_4F077C4 == 2)
    && (unsigned int)sub_8D2600(v7)
    && ((*(_BYTE *)(v7 + 140) & 0xFB) != 8 || !(unsigned int)sub_8D4C10(v7, dword_4F077C4 != 2)) )
  {
    v11 = (__int64 *)sub_73E830(a1);
    sub_6E70E0(v11, (__int64)a4);
  }
  else
  {
    v8 = (__int64 *)sub_731250(a1);
    sub_6E7150(v8, (__int64)a4);
    v9 = dword_4F077C4 == 2;
    a4[11] = a5;
    if ( v9 && (unsigned int)sub_8D32E0(v7) )
      sub_6F82C0((__int64)a4, (__int64)a4, v12, v13, v14, v15);
  }
  *(_QWORD *)((char *)a4 + 68) = *a2;
  *(_QWORD *)((char *)a4 + 76) = *a3;
  return sub_6E3280((__int64)a4, 0);
}
