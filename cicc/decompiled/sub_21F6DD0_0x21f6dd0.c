// Function: sub_21F6DD0
// Address: 0x21f6dd0
//
__int64 __fastcall sub_21F6DD0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 result; // rax

  v4 = *a2;
  v5 = (a2[1] - *a2) >> 3;
  if ( !(_DWORD)v5 )
    return 0;
  v6 = 0;
  v7 = 8LL * (unsigned int)(v5 - 1);
  while ( 1 )
  {
    result = sub_21F6CA0(a1, *(_QWORD *)(v4 + v6), a3, *(_QWORD *)(a1 + 208));
    if ( (_BYTE)result )
      break;
    if ( v6 == v7 )
      return 0;
    v4 = *a2;
    v6 += 8;
  }
  return result;
}
