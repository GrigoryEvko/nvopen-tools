// Function: sub_33CE520
// Address: 0x33ce520
//
__int64 __fastcall sub_33CE520(__int64 a1, __int64 a2, __int64 a3, const void **a4)
{
  bool v6; // cc
  unsigned __int64 *v7; // r13
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // r12
  unsigned int v11; // ebx
  unsigned __int64 v12; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-58h]
  unsigned __int64 v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-48h]
  unsigned __int64 v16; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-38h]

  v6 = *((_DWORD *)a4 + 2) <= 0x40u;
  v13 = 1;
  v12 = 0;
  if ( v6 )
  {
    LODWORD(v7) = 1;
    if ( (unsigned int)sub_39FAC40(*a4) == 1 )
      return (unsigned int)v7;
  }
  else
  {
    LODWORD(v7) = 1;
    if ( (unsigned int)sub_C44630((__int64)a4) == 1 )
      return (unsigned int)v7;
  }
  LODWORD(v7) = sub_33CD9D0(*(_QWORD *)(a1 + 8), a2, a3, (__int64)a4, (__int64)&v12, **(_DWORD **)a1 + 1);
  if ( (_BYTE)v7 )
  {
    v15 = *((_DWORD *)a4 + 2);
    if ( v15 > 0x40 )
    {
      v7 = &v14;
      sub_C43780((__int64)&v14, a4);
      if ( v15 > 0x40 )
      {
        sub_C43B90(&v14, (__int64 *)&v12);
        v11 = v15;
        v10 = v14;
        v15 = 0;
        v17 = v11;
        v16 = v14;
        if ( v11 > 0x40 )
        {
          LOBYTE(v7) = v11 == (unsigned int)sub_C444A0((__int64)&v16);
          if ( v10 )
          {
            j_j___libc_free_0_0(v10);
            if ( v15 > 0x40 )
            {
              if ( v14 )
                j_j___libc_free_0_0(v14);
            }
          }
          goto LABEL_6;
        }
        goto LABEL_12;
      }
      v9 = v14;
    }
    else
    {
      v9 = (unsigned __int64)*a4;
    }
    v10 = v12 & v9;
LABEL_12:
    LOBYTE(v7) = v10 == 0;
  }
LABEL_6:
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  return (unsigned int)v7;
}
