// Function: sub_72FC40
// Address: 0x72fc40
//
__int64 __fastcall sub_72FC40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 *v4; // rdx
  __int64 *v5; // rcx
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // rax

  sub_72EE40(a1, 7u, a2);
  v2 = *(int *)(a2 + 240);
  if ( (_DWORD)v2 == -1 )
  {
    v4 = (__int64 *)(a2 + 120);
    if ( *(_BYTE *)(a1 + 136) == 2 )
      v4 = (__int64 *)(a2 + 112);
    goto LABEL_20;
  }
  v3 = qword_4F04C68[0] + 776 * v2;
  if ( *(_BYTE *)(a1 + 136) != 2 )
  {
    v4 = (__int64 *)(a2 + 120);
    if ( v3 )
    {
      v5 = (__int64 *)(v3 + 288);
      result = *(_QWORD *)(v3 + 288);
      goto LABEL_5;
    }
LABEL_20:
    result = *v4;
    if ( !*v4 )
    {
      *(_QWORD *)(a1 + 112) = 0;
      *v4 = a1;
      return result;
    }
    v5 = 0;
    goto LABEL_10;
  }
  v4 = (__int64 *)(a2 + 112);
  if ( !v3 )
    goto LABEL_20;
  v7 = *(_QWORD *)(v3 + 24);
  v8 = v3 + 32;
  if ( !v7 )
    v7 = v8;
  result = *(_QWORD *)(v7 + 40);
  v5 = (__int64 *)(v7 + 40);
LABEL_5:
  if ( result && !*(_QWORD *)(result + 8) )
  {
    *(_QWORD *)(result + 112) = a1;
    *(_QWORD *)(a1 + 112) = 0;
    *v5 = a1;
    return result;
  }
  result = *v4;
  if ( !*v4 )
  {
    *(_QWORD *)(a1 + 112) = 0;
    *v4 = a1;
LABEL_12:
    if ( !*(_QWORD *)(a1 + 112) )
      *v5 = a1;
    return result;
  }
  do
  {
LABEL_10:
    if ( *(_QWORD *)(result + 8) )
      break;
    v4 = (__int64 *)(result + 112);
    result = *(_QWORD *)(result + 112);
  }
  while ( result );
  *(_QWORD *)(a1 + 112) = result;
  *v4 = a1;
  if ( v5 )
    goto LABEL_12;
  return result;
}
