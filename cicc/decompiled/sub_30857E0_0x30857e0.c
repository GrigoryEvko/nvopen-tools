// Function: sub_30857E0
// Address: 0x30857e0
//
__int64 __fastcall sub_30857E0(__int64 a1, unsigned __int8 *a2, unsigned __int8 **a3, __int64 a4)
{
  unsigned int v6; // r15d
  __int64 *v8; // rbx
  __int16 v9; // ax
  unsigned __int8 *v10; // rdx
  __int64 *v11; // rax
  unsigned int v12; // eax
  unsigned __int8 *v13; // rbx
  unsigned __int8 *v14; // [rsp+8h] [rbp-38h]

  if ( *a2 == 22 )
  {
    if ( byte_502D428[0] || (unsigned __int8)sub_B2D700((__int64)a2) )
    {
      *a3 = a2;
      return 1;
    }
    *a3 = a2;
  }
  v8 = sub_DD8400(a4, (__int64)a2);
  if ( *((_WORD *)v8 + 12) == 8 )
  {
    do
    {
      v8 = *(__int64 **)v8[4];
      v9 = *((_WORD *)v8 + 12);
      if ( v9 == 15 )
      {
        v10 = (unsigned __int8 *)*(v8 - 1);
        if ( *v10 != 22 )
          break;
        v6 = byte_502D428[0];
        if ( byte_502D428[0] )
        {
          *a3 = v10;
          return v6;
        }
        v14 = (unsigned __int8 *)*(v8 - 1);
        v6 = sub_B2D700((__int64)v10);
        *a3 = v14;
        if ( (_BYTE)v6 )
          return v6;
        v9 = *((_WORD *)v8 + 12);
      }
      if ( v9 == 5 )
      {
        if ( (unsigned __int8)sub_3085060((__int64)v8, a3) )
          return 1;
        v9 = *((_WORD *)v8 + 12);
      }
    }
    while ( v9 == 8 );
  }
  v11 = sub_DD8400(a4, (__int64)a2);
  if ( *((_WORD *)v11 + 12) == 5 && (unsigned __int8)sub_3085060((__int64)v11, a3) )
  {
    return 1;
  }
  else
  {
    LOBYTE(v12) = sub_CEFF50((__int64)a2);
    v6 = v12;
    if ( (_BYTE)v12 && (v13 = sub_98ACB0(a2, 6u), v13 == sub_98ACB0(v13, 6u)) )
      *a3 = v13;
    else
      return 0;
  }
  return v6;
}
