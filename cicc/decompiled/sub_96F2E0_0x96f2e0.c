// Function: sub_96F2E0
// Address: 0x96f2e0
//
__int64 __fastcall sub_96F2E0(unsigned int a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v10; // rax
  _BYTE *v11; // r14
  __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int8 v14; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v19; // [rsp+8h] [rbp-38h]

  if ( a1 - 13 > 0x11 )
    return sub_96E6C0(a1, a2, a3, a4);
  v10 = sub_96ED60(a2, a5, 0);
  if ( !v10 )
    return 0;
  v19 = v10;
  v11 = (_BYTE *)sub_96ED60((__int64)a3, a5, 0);
  if ( !v11 )
    return 0;
  v12 = v19;
  if ( !a6 )
  {
    v13 = sub_96F1D0(a5);
    v12 = v19;
    if ( v13 )
    {
      v14 = *(_BYTE *)(v13 + 1);
      if ( (v14 & 2) != 0 || ((v14 >> 1) & 0x38) != 0 )
        return 0;
    }
  }
  v15 = sub_96E6C0(a1, v12, v11, a4);
  if ( !v15 )
    return 0;
  v16 = sub_96ED60(v15, a5, 1);
  v17 = v16;
  if ( !v16 || !a6 && (unsigned __int8)sub_AD8220(v16) )
    return 0;
  return v17;
}
