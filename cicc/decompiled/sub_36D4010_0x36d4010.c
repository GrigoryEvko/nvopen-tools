// Function: sub_36D4010
// Address: 0x36d4010
//
__int64 __fastcall sub_36D4010(_QWORD *a1, __int64 a2, unsigned __int64 a3, char a4)
{
  _BYTE *v6; // rax
  __int64 v7; // r12
  __int64 result; // rax
  char v9; // r14
  unsigned int v10; // r15d
  bool v11; // zf
  const char *v12; // r13
  _QWORD *v13; // rdi
  __int64 *v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // r13
  _QWORD v17[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v18; // [rsp+20h] [rbp-40h]

  v6 = sub_BA8CD0((__int64)a1, a2, a3, 0);
  if ( !v6 )
    return 0;
  v7 = (__int64)v6;
  if ( sub_B2FC80((__int64)v6) )
    return 0;
  v9 = a4;
  result = sub_36D31C0((__int64)a1, v7, a4);
  if ( !(_BYTE)result )
    return 0;
  v10 = (unsigned __int8)qword_5040C68;
  if ( (_BYTE)qword_5040C68 )
  {
    v11 = a4 == 0;
    v12 = "nvptx$device$init";
    if ( v11 )
      v12 = "nvptx$device$fini";
    if ( !sub_BA8CB0((__int64)a1, (__int64)v12, 0x11u) )
    {
      v13 = (_QWORD *)*a1;
      v17[0] = v12;
      v18 = 261;
      v17[1] = 17;
      v14 = (__int64 *)sub_BCB120(v13);
      v15 = sub_BCF640(v14, 0);
      v16 = sub_B2CE20(v15, 5, 0, (__int64)v17, (__int64)a1);
      sub_B2CD60(v16, "nvvm.maxclusterrank", 0x13u, "1", 1u);
      sub_B2CD60(v16, "nvvm.maxntid", 0xCu, "1", 1u);
      *(_WORD *)(v16 + 2) = *(_WORD *)(v16 + 2) & 0xC00F | 0x470;
      sub_36D1960((_QWORD *)v16, v9);
      sub_B30290(v7);
      return v10;
    }
    return 0;
  }
  return result;
}
