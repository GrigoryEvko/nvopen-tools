// Function: sub_F03F40
// Address: 0xf03f40
//
__int64 __fastcall sub_F03F40(__int64 a1)
{
  _DWORD *v1; // rdx

  v1 = *(_DWORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v1 <= 3u )
    return sub_CB6200(a1, (unsigned __int8 *)"None", 4u);
  *v1 = 1701736270;
  *(_QWORD *)(a1 + 32) += 4LL;
  return a1;
}
