// Function: sub_68DF50
// Address: 0x68df50
//
__int64 __fastcall sub_68DF50(__int64 a1, unsigned __int64 a2, _DWORD *a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v7; // rdi
  unsigned __int64 v8; // r12
  _DWORD *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // r8
  unsigned __int64 v22; // rdi
  _DWORD *v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  _DWORD *v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  char v34; // al
  __int64 v35; // [rsp+8h] [rbp-38h]

  *a5 = 0;
  if ( a2 )
  {
    v7 = *(_QWORD *)a2;
    v8 = a2;
    if ( *(_QWORD *)a2 )
    {
      v9 = (_DWORD *)sub_6E1A20(v7);
      if ( (unsigned int)sub_6E5430(v7, a2, v10, v11, v12, v13) )
        sub_6851C0(0x8Cu, v9);
      return sub_6EB5C0(a1);
    }
    sub_6E65B0(a2);
    v16 = *(_QWORD *)(a2 + 24);
    v17 = v16 + 8;
    if ( *(_BYTE *)(v16 + 25) == 1 )
      sub_6FA3A0(v16 + 8);
    v18 = *(_QWORD *)(v16 + 8);
    if ( (unsigned int)sub_8DBE70(v18) )
      goto LABEL_16;
    if ( (unsigned int)sub_8D2E30(v18) )
    {
      v35 = sub_8D46C0(v18);
      v20 = sub_8D2600(v35);
      v21 = v35;
      if ( !v20 )
      {
        if ( (*(_BYTE *)(v18 + 140) & 0xFB) == 8
          && (a2 = dword_4F077C4 != 2, v34 = sub_8D4C10(v18, a2), v21 = v35, (v34 & 1) != 0)
          || (*(_BYTE *)(v21 + 140) & 0xFB) == 8 && (a2 = dword_4F077C4 != 2, (sub_8D4C10(v21, a2) & 1) != 0) )
        {
          v22 = v8;
          v23 = (_DWORD *)sub_6E1A20(v8);
          if ( (unsigned int)sub_6E5430(v22, a2, v24, v25, v26, v27) )
            sub_6851C0(0xC2Eu, v23);
          return sub_6EB5C0(a1);
        }
LABEL_16:
        *a5 = sub_6F7150(v17, a2, v19);
        return sub_6EB5C0(a1);
      }
    }
    v28 = (_DWORD *)sub_6E1A20(a2);
    if ( (unsigned int)sub_6E5430(a2, a2, v29, v30, v31, v32) )
      sub_6851C0(0x354u, v28);
  }
  else if ( (unsigned int)sub_6E5430(a1, 0, a3, a4, a5, a6) )
  {
    sub_6851C0(0xA5u, a3);
  }
  return sub_6EB5C0(a1);
}
