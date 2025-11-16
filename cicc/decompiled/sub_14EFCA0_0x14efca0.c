// Function: sub_14EFCA0
// Address: 0x14efca0
//
void __fastcall sub_14EFCA0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax

  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 == *(_QWORD *)(a1 + 16) )
  {
    sub_14EFA10((char **)a1, (char *)v3, a2);
  }
  else
  {
    if ( v3 )
    {
      v4 = *a2;
      *(_QWORD *)v3 = 6;
      *(_QWORD *)(v3 + 8) = 0;
      *(_QWORD *)(v3 + 16) = v4;
      if ( v4 != 0 && v4 != -8 && v4 != -16 )
        sub_164C220(v3);
      v3 = *(_QWORD *)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v3 + 24;
  }
}
