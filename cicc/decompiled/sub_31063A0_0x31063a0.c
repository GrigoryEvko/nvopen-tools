// Function: sub_31063A0
// Address: 0x31063a0
//
__int64 __fastcall sub_31063A0(_BYTE *a1, __int64 a2, unsigned __int8 *a3)
{
  int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v8; // rax

  if ( a3 )
  {
    if ( (*a1 || (unsigned int)*a3 - 30 > 0xA) && (unsigned __int8)sub_98CD80((char *)a3) )
    {
      if ( (unsigned int)*a3 - 30 > 0xA )
      {
        v6 = *((_QWORD *)a3 + 4);
        if ( v6 != *((_QWORD *)a3 + 5) + 48LL )
        {
LABEL_12:
          if ( v6 )
            return v6 - 24;
        }
      }
      else
      {
        v4 = sub_B46E30((__int64)a3);
        if ( v4 )
        {
          if ( v4 == 1 )
          {
            v6 = *(_QWORD *)(sub_B46EC0((__int64)a3, 0) + 56);
            if ( v6 )
              return v6 - 24;
            return 0;
          }
          v8 = sub_3105700((__int64)a1, *((_QWORD *)a3 + 5), v5);
          if ( v8 )
          {
            v6 = *(_QWORD *)(v8 + 56);
            goto LABEL_12;
          }
        }
      }
    }
    return 0;
  }
  return 0;
}
