// Function: sub_394AEC0
// Address: 0x394aec0
//
char __fastcall sub_394AEC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rbx
  char result; // al
  int v6; // edx
  __int64 v7; // rax
  _QWORD *v8; // r12

  v4 = (_QWORD *)a1;
  result = sub_1593BB0(a1, a2, a3, a4);
  if ( !result )
  {
    v6 = *(unsigned __int8 *)(a1 + 16);
    if ( (_BYTE)v6 == 9 )
      return 1;
    if ( (unsigned int)(v6 - 6) > 2 )
      return result;
    v7 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v8 = *(_QWORD **)(a1 - 8);
      v4 = &v8[v7];
    }
    else
    {
      v8 = (_QWORD *)(a1 - v7 * 8);
    }
    if ( v4 == v8 )
      return 1;
    while ( 1 )
    {
      result = sub_394AEC0(*v8);
      if ( !result )
        break;
      v8 += 3;
      if ( v4 == v8 )
        return 1;
    }
  }
  return result;
}
