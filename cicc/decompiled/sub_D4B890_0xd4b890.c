// Function: sub_D4B890
// Address: 0xd4b890
//
unsigned __int64 __fastcall sub_D4B890(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rsi
  __int64 v3; // r13
  __int64 v4; // rdi
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 v7; // r14

  if ( !(unsigned __int8)sub_D4B3D0(a1) )
    return 0;
  v1 = sub_D4B130(a1);
  v2 = sub_D47930(a1);
  if ( !v2 )
    return 0;
  if ( !(unsigned __int8)sub_D46CA0(a1, v2) )
    return 0;
  v3 = sub_D47600(a1);
  if ( !v3 )
    return 0;
  v4 = sub_AA5510(v1);
  if ( !v4 )
    return 0;
  v5 = sub_986580(v4);
  v6 = v5;
  if ( *(_BYTE *)v5 != 31 || (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) == 1 )
    return 0;
  v7 = *(_QWORD *)(v5 - 32);
  if ( v1 == v7 )
    v7 = *(_QWORD *)(v5 - 64);
  if ( v7 != sub_D52390(v3, v7, 1) )
    return 0;
  return v6;
}
