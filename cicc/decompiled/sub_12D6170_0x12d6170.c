// Function: sub_12D6170
// Address: 0x12d6170
//
__int64 __fastcall sub_12D6170(__int64 a1, unsigned int a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  _QWORD *v6; // r15
  __int64 v7; // r14
  _QWORD *v9; // rbx
  __int64 v10; // rax
  _DWORD v11[13]; // [rsp+Ch] [rbp-34h] BYREF

  v11[0] = a2;
  v2 = sub_168FA50(a1, v11, 1);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = (_QWORD *)(v3 + 8LL * (unsigned int)v2);
  v5 = (_QWORD *)(v3 + 8 * HIDWORD(v2));
  if ( v4 == v5 )
    return 0;
  while ( 1 )
  {
    v6 = v4;
    if ( *v4 )
    {
      if ( a2 && (unsigned __int8)sub_1690410(*v4, a2) )
        break;
    }
    if ( v5 == ++v4 )
      return 0;
  }
  if ( v5 == v4 )
    return 0;
  do
  {
    v7 = *v6;
    v9 = v6 + 1;
    v10 = *(_QWORD *)(*v6 + 16LL);
    if ( !v10 )
      v10 = *v6;
    *(_BYTE *)(v10 + 44) |= 1u;
    if ( v9 == v5 )
      break;
    while ( 1 )
    {
      v6 = v9;
      if ( *v9 )
      {
        if ( (unsigned __int8)sub_1690410(*v9, a2) )
          break;
      }
      if ( v5 == ++v9 )
        return v7;
    }
  }
  while ( v5 != v9 );
  return v7;
}
