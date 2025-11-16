// Function: sub_3176D40
// Address: 0x3176d40
//
__int64 __fastcall sub_3176D40(__int64 *a1, _QWORD *a2)
{
  _BYTE *v2; // r13
  unsigned __int64 v3; // rax
  _BYTE *v4; // rax
  __int64 v6; // rdx
  _BYTE *v7; // r14
  _BYTE *v8; // rbx
  signed __int64 v9; // rax
  _BYTE *v10; // r12
  unsigned __int8 *v11; // rbx
  unsigned __int8 *v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  _BYTE *v15; // [rsp+0h] [rbp-40h] BYREF
  _BYTE *v16; // [rsp+8h] [rbp-38h]

  if ( !a2[2] )
    goto LABEL_12;
  v2 = (_BYTE *)a2[1];
  v3 = (unsigned __int8)v2[8];
  if ( (_BYTE)v3 == 14 )
  {
LABEL_7:
    if ( !(unsigned __int8)sub_B2D680((__int64)a2) )
      goto LABEL_8;
LABEL_11:
    if ( (unsigned __int8)sub_B2DCE0(a2[3]) )
      goto LABEL_8;
LABEL_12:
    LODWORD(v2) = 0;
    return (unsigned int)v2;
  }
  if ( !(_BYTE)qword_50344A8 )
    goto LABEL_12;
  if ( (unsigned __int8)v3 > 0xCu || (v6 = 4143, !_bittest64(&v6, v3)) )
  {
    if ( (v3 & 0xFD) != 4 && (_BYTE)v3 != 15 )
      goto LABEL_12;
    goto LABEL_7;
  }
  if ( (unsigned __int8)sub_B2D680((__int64)a2) )
    goto LABEL_11;
LABEL_8:
  if ( (unsigned __int8)sub_2A641A0((__int64 *)*a1, a2[3]) )
  {
    if ( v2[8] != 15 )
    {
      v4 = (_BYTE *)sub_2A64F10(*a1, (__int64)a2);
      return sub_2A62E90(v4);
    }
    sub_2A65F10((unsigned __int64 *)&v15, (__int64 *)*a1, (__int64)a2);
    v7 = v16;
    v8 = v15;
    v9 = 0xCCCCCCCCCCCCCCCDLL * ((v16 - v15) >> 3);
    if ( v9 >> 2 > 0 )
    {
      v2 = &v15[160 * (v9 >> 2)];
      while ( !(unsigned __int8)sub_2A62E90(v8) )
      {
        v10 = v8 + 40;
        if ( (unsigned __int8)sub_2A62E90(v8 + 40)
          || (v10 = v8 + 80, (unsigned __int8)sub_2A62E90(v8 + 80))
          || (v10 = v8 + 120, (unsigned __int8)sub_2A62E90(v8 + 120)) )
        {
          LOBYTE(v2) = v7 != v10;
          goto LABEL_27;
        }
        v8 += 160;
        if ( v2 == v8 )
        {
          v9 = 0xCCCCCCCCCCCCCCCDLL * ((v7 - v8) >> 3);
          goto LABEL_41;
        }
      }
      goto LABEL_24;
    }
LABEL_41:
    if ( v9 != 2 )
    {
      if ( v9 != 3 )
      {
        if ( v9 != 1 )
        {
LABEL_44:
          LODWORD(v2) = 0;
          goto LABEL_27;
        }
LABEL_49:
        if ( !(unsigned __int8)sub_2A62E90(v8) )
          goto LABEL_44;
LABEL_24:
        LOBYTE(v2) = v7 != v8;
LABEL_27:
        v11 = v16;
        v12 = v15;
        if ( v16 != v15 )
        {
          do
          {
            if ( (unsigned int)*v12 - 4 <= 1 )
            {
              if ( *((_DWORD *)v12 + 8) > 0x40u )
              {
                v13 = *((_QWORD *)v12 + 3);
                if ( v13 )
                  j_j___libc_free_0_0(v13);
              }
              if ( *((_DWORD *)v12 + 4) > 0x40u )
              {
                v14 = *((_QWORD *)v12 + 1);
                if ( v14 )
                  j_j___libc_free_0_0(v14);
              }
            }
            v12 += 40;
          }
          while ( v11 != v12 );
          v12 = v15;
        }
        if ( v12 )
          j_j___libc_free_0((unsigned __int64)v12);
        return (unsigned int)v2;
      }
      if ( (unsigned __int8)sub_2A62E90(v8) )
        goto LABEL_24;
      v8 += 40;
    }
    if ( (unsigned __int8)sub_2A62E90(v8) )
      goto LABEL_24;
    v8 += 40;
    goto LABEL_49;
  }
  return 1;
}
