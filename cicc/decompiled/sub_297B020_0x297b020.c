// Function: sub_297B020
// Address: 0x297b020
//
__int64 __fastcall sub_297B020(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rcx
  __int64 v6; // r8
  _BYTE *v7; // r9
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rsi

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a2 + 48 )
    goto LABEL_31;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_31:
    BUG();
  if ( *(_BYTE *)(v2 - 24) != 31 || (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v3 = *(_QWORD *)(v2 - 56);
  v4 = *(_QWORD *)(v2 - 88);
  if ( a2 != v4 && a2 != v3 && v4 != v3 )
  {
    if ( sub_AA54C0(v3) && v4 == sub_AA56F0(v3) )
      goto LABEL_29;
    if ( sub_AA54C0(v4) && v3 == sub_AA56F0(v4) )
      goto LABEL_25;
    if ( !sub_AA54C0(v3) )
      return 0;
    if ( !sub_AA54C0(v4) )
      return 0;
    if ( !sub_AA56F0(v4) )
      return 0;
    if ( a2 == sub_AA56F0(v4) )
      return 0;
    v8 = sub_AA56F0(v4);
    if ( v8 != sub_AA56F0(v3) )
      return 0;
    v9 = *(_QWORD *)(v4 + 56);
    v5 = v4 + 48;
    if ( v4 + 48 != v9 )
    {
      v10 = 0;
      do
      {
        v9 = *(_QWORD *)(v9 + 8);
        ++v10;
      }
      while ( v5 != v9 );
      if ( v10 == 1 )
      {
LABEL_29:
        v13 = a2;
        v14 = v3;
        return sub_297A990(a1, v14, v13, v5, v6, v7);
      }
    }
    v11 = *(_QWORD *)(v3 + 56);
    v5 = v3 + 48;
    if ( v3 + 48 != v11 )
    {
      v12 = 0;
      do
      {
        v11 = *(_QWORD *)(v11 + 8);
        ++v12;
      }
      while ( v5 != v11 );
      if ( v12 == 1 )
      {
LABEL_25:
        v13 = a2;
        v14 = v4;
        return sub_297A990(a1, v14, v13, v5, v6, v7);
      }
    }
  }
  return 0;
}
