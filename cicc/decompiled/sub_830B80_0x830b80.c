// Function: sub_830B80
// Address: 0x830b80
//
__int64 __fastcall sub_830B80(__int64 *a1, __int64 a2, int a3, __int64 *a4, _QWORD *a5, _QWORD *a6)
{
  __int64 *v10; // rax
  __int64 result; // rax
  _QWORD *v12; // rax
  __int64 **v13; // r15
  int v14; // eax
  __int64 *v15; // rax
  _QWORD *v16; // rax

  if ( (unsigned int)sub_693580() )
  {
    v13 = sub_5F7420(a1, a4, 0, 1);
    if ( v13 )
    {
      v14 = sub_85E8D0();
      v15 = sub_830AC0((__int64)v13, v14, 0);
      sub_6E70E0(v15, (__int64)a6);
    }
    else if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
    {
      if ( (unsigned int)sub_6E5430() )
        sub_6851C0(0x6CAu, a4);
      sub_6E6260(a6);
    }
    else
    {
      v16 = sub_726700(24);
      v16[7] = 0;
      *v16 = a2;
      *(_QWORD *)((char *)v16 + 28) = *a4;
      *((_BYTE *)v16 + 27) = (2 * (a3 & 1)) | *((_BYTE *)v16 + 27) & 0xFD;
      sub_6E70E0(v16, (__int64)a6);
    }
  }
  else if ( a1 )
  {
    v10 = sub_73E830((__int64)a1);
    sub_6E70E0(v10, (__int64)a6);
  }
  else
  {
    v12 = sub_726700(24);
    v12[7] = 0;
    *v12 = a2;
    *(_QWORD *)((char *)v12 + 28) = *a4;
    *((_BYTE *)v12 + 27) = (2 * (a3 & 1)) | *((_BYTE *)v12 + 27) & 0xFD;
    sub_6E70E0(v12, (__int64)a6);
    if ( !(unsigned int)sub_830310(0, 0, 0, a4) )
      sub_721090();
  }
  *(_QWORD *)((char *)a6 + 68) = *a4;
  *(_QWORD *)((char *)a6 + 76) = *a5;
  if ( !a3 )
    sub_6E3280((__int64)a6, 0);
  sub_6E26D0(2, (__int64)a6);
  result = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 20LL) |= 0x20u;
  return result;
}
