// Function: sub_2840260
// Address: 0x2840260
//
unsigned __int64 __fastcall sub_2840260(__int64 *a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r14
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  int v15; // edx
  unsigned __int64 result; // rax

  v4 = *a4;
  if ( !sub_DADE90(a1[2], *a4, a1[5]) )
    return a3;
  v9 = a1[7];
  v10 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 == v9 + 48 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    v11 = *(unsigned __int8 *)(v10 - 24);
    v12 = v10 - 24;
    if ( (unsigned int)(v11 - 30) >= 0xB )
      v12 = 0;
  }
  if ( !(unsigned __int8)sub_F80650(a2, v4, v12, v6, v7, v8) )
    return a3;
  v13 = a1[7];
  v14 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v14 == v13 + 48 )
    return 0;
  if ( !v14 )
    BUG();
  v15 = *(unsigned __int8 *)(v14 - 24);
  result = v14 - 24;
  if ( (unsigned int)(v15 - 30) >= 0xB )
    return 0;
  return result;
}
