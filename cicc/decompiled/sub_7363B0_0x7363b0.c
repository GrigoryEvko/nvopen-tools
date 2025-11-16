// Function: sub_7363B0
// Address: 0x7363b0
//
__int64 __fastcall sub_7363B0(__int64 a1, int a2, __int64 a3)
{
  _BYTE *v3; // rdx
  _QWORD *v4; // rax
  __int64 result; // rax
  _QWORD *v6; // rdx
  __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  v3 = sub_735B90(a2, a3, &v7);
  v4 = (_QWORD *)*((_QWORD *)v3 + 29);
  if ( v4 )
  {
    if ( v7 )
    {
      **(_QWORD **)(v7 + 96) = a1;
    }
    else
    {
      do
      {
        v6 = v4;
        v4 = (_QWORD *)*v4;
      }
      while ( v4 );
      *v6 = a1;
    }
  }
  else
  {
    *((_QWORD *)v3 + 29) = a1;
  }
  result = v7;
  if ( v7 )
    *(_QWORD *)(v7 + 96) = a1;
  return result;
}
