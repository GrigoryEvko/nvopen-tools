// Function: sub_13100A0
// Address: 0x13100a0
//
__int64 __fastcall sub_13100A0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 a5, _BYTE *a6)
{
  __int64 v6; // r13
  __int64 v9; // r14
  __int64 v10; // r8
  __int64 v11; // rdx

  v6 = (unsigned int)a5;
  v9 = *a3;
  sub_1316790(
    a1,
    a2,
    a4,
    unk_5060A20 + 2LL * (unsigned int)a5,
    a5,
    (int)*(unsigned __int16 *)(unk_5060A20 + 2LL * (unsigned int)a5) >> *(_BYTE *)(*a3 + (unsigned int)a5 + 52));
  *(_BYTE *)(v9 + v6 + 88) = 1;
  v10 = *(_QWORD *)*a4;
  v11 = *a4 + 8;
  if ( (unsigned __int16)*a4 == *((_WORD *)a4 + 8) )
  {
    if ( (unsigned __int16)*a4 == *((_WORD *)a4 + 10) )
    {
      *a6 = 0;
      return 0;
    }
    else
    {
      *a4 = v11;
      *((_WORD *)a4 + 8) = v11;
      *a6 = 1;
    }
  }
  else
  {
    *a4 = v11;
    *a6 = 1;
  }
  return v10;
}
