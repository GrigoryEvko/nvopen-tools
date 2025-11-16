// Function: sub_30200E0
// Address: 0x30200e0
//
__int64 __fastcall sub_30200E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _BYTE *v4; // rdi
  __int64 v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // rdx

  v2 = *(_QWORD *)(a1 + 16);
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v4 = *(_BYTE **)(v2 + 24);
    if ( *v4 <= 0x15u )
    {
      if ( (unsigned __int8)sub_30200E0(v4, a2) )
        return 1;
      goto LABEL_4;
    }
    if ( *v4 <= 0x1Cu )
      goto LABEL_4;
    v5 = sub_B43CB0((__int64)v4);
    if ( !v5 )
      goto LABEL_4;
    if ( *(_BYTE *)(a2 + 28) )
      break;
    if ( sub_C8CA60(a2, v5) )
      return 1;
LABEL_4:
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 0;
  }
  v6 = *(_QWORD **)(a2 + 8);
  v7 = &v6[*(unsigned int *)(a2 + 20)];
  if ( v6 == v7 )
    goto LABEL_4;
  while ( v5 != *v6 )
  {
    if ( v7 == ++v6 )
      goto LABEL_4;
  }
  return 1;
}
