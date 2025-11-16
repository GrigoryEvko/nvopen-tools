// Function: sub_6E3360
// Address: 0x6e3360
//
__int64 __fastcall sub_6E3360(
        __int64 a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64, _QWORD, _DWORD *, _QWORD),
        _DWORD *a4,
        __int64 a5,
        __int64 a6)
{
  unsigned int v6; // r13d
  __int64 v7; // r12
  _QWORD *i; // r15
  __int64 v10; // r14
  unsigned __int8 v11; // bl
  __int64 v12; // rdi
  __int64 v13; // r12
  unsigned int v14; // r13d
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // r13d
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-130h]
  __int64 v22; // [rsp+8h] [rbp-128h]
  unsigned int v23; // [rsp+14h] [rbp-11Ch]
  _QWORD v26[10]; // [rsp+30h] [rbp-100h] BYREF
  int v27; // [rsp+80h] [rbp-B0h]
  int v28; // [rsp+90h] [rbp-A0h]

  v6 = a2;
  v7 = a1;
  v23 = a5;
  *a4 = 0;
  if ( (*(_BYTE *)(a1 + 25) & 3) != 0 )
  {
    for ( i = (_QWORD *)a1; ; i = (_QWORD *)v7 )
    {
      sub_76C7C0(v26, a2, a3, a4, a5, a6);
      v26[0] = sub_6DF820;
      v28 = 1;
      sub_76CDC0(v7);
      if ( !v27 )
        return a3(v7, v6, a4, v23);
      v10 = *(_QWORD *)(v7 + 72);
      v11 = *(_BYTE *)(v7 + 56);
      v7 = *(_QWORD *)(v10 + 16);
      if ( (unsigned int)sub_6DEAC0((__int64)i) )
        break;
      if ( (unsigned __int8)(v11 - 103) <= 1u )
      {
        v21 = *(_QWORD *)(v7 + 16);
        v14 = sub_731770(i, 0) | v6;
        v22 = a3(v10, v14, a4, v23);
        if ( unk_4D03F94 && v11 == 103 )
        {
          sub_7F0F50(v10);
          sub_7F0F50(v22);
        }
        v15 = sub_6E3360(v7, v14, a3, v26, v23);
        if ( LODWORD(v26[0]) )
          *a4 = 1;
        v16 = sub_6E3360(v21, v14, a3, v26, v23);
        if ( LODWORD(v26[0]) )
          *a4 = 1;
        *(_QWORD *)(v22 + 16) = v15;
        *(_QWORD *)(v15 + 16) = v16;
        v17 = sub_73DC30(v11, *i, v22);
        *(_BYTE *)(v17 + 58) |= 1u;
        v13 = v17;
        sub_730580(i, v17);
        return v13;
      }
      if ( v11 != 91 )
      {
        if ( !dword_4F077BC || (unsigned __int8)(v11 - 71) > 1u )
          sub_721090(i);
        v18 = sub_731770(i, 0) | v6;
        v19 = sub_6E3360(v10, v18, a3, v26, v23);
        if ( LODWORD(v26[0]) )
          *a4 = 1;
        v20 = sub_6E3360(v7, v18, a3, v26, v23);
        if ( LODWORD(v26[0]) )
          *a4 = 1;
        *(_QWORD *)(v19 + 16) = v20;
        return sub_73DC30(v11, *i, v19);
      }
      a2 = 0;
      v6 |= sub_731770(i, 0);
      *a4 = 0;
      if ( (*(_BYTE *)(v7 + 25) & 3) == 0 )
        return a3(v7, v6, a4, v23);
    }
    if ( v11 == 94 )
      v12 = sub_6E3360(v10, v6, a3, a4, v23);
    else
      v12 = a3(v10, v6, a4, v23);
    v13 = sub_73DE50(v12, *(_QWORD *)(v7 + 56));
    sub_730580(i, v13);
    return v13;
  }
  return a3(v7, v6, a4, v23);
}
