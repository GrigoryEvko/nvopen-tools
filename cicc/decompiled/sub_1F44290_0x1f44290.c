// Function: sub_1F44290
// Address: 0x1f44290
//
bool __fastcall sub_1F44290(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v6; // r15
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  __int64 v9; // rbx
  bool result; // al
  int v11; // eax

  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL) + 112LL;
  if ( (unsigned __int8)sub_1560180(v6, 34) || (unsigned __int8)sub_1560180(v6, 17) )
  {
    v7 = sub_1F44230(a1, 1);
    v8 = 0xFFFFFFFFLL;
    v9 = v7;
  }
  else
  {
    v9 = (unsigned int)sub_1F44230(a1, 0);
    v11 = sub_1F44280(a1);
    v8 = 0xFFFFFFFFLL;
    if ( v11 )
      v8 = (unsigned int)sub_1F44280(a1);
  }
  result = 0;
  if ( a4 <= v8 )
    return 100 * a3 >= a4 * v9;
  return result;
}
