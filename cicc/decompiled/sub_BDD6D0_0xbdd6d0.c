// Function: sub_BDD6D0
// Address: 0xbdd6d0
//
__int64 __fastcall sub_BDD6D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax
  __int64 result; // rax

  v2 = *a1;
  if ( *a1 )
  {
    sub_CA0E80(a2, v2);
    v3 = *(_BYTE **)(v2 + 32);
    if ( (unsigned __int64)v3 >= *(_QWORD *)(v2 + 24) )
    {
      sub_CB5D20(v2, 10);
    }
    else
    {
      *(_QWORD *)(v2 + 32) = v3 + 1;
      *v3 = 10;
    }
  }
  result = *((unsigned __int8 *)a1 + 154);
  *((_BYTE *)a1 + 153) = 1;
  *((_BYTE *)a1 + 152) |= result;
  return result;
}
