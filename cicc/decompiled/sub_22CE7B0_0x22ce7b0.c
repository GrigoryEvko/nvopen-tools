// Function: sub_22CE7B0
// Address: 0x22ce7b0
//
unsigned __int8 *__fastcall sub_22CE7B0(unsigned __int8 *a1, unsigned __int64 a2, __int64 *a3)
{
  __int64 v3; // r15
  __int64 v4; // r13
  __int64 v5; // rax
  int v7; // [rsp+4h] [rbp-9Ch]
  __int64 *v8; // [rsp+8h] [rbp-98h]
  _OWORD v9[2]; // [rsp+10h] [rbp-90h] BYREF
  __int128 v10; // [rsp+30h] [rbp-70h]
  unsigned __int8 v11[40]; // [rsp+40h] [rbp-60h] BYREF
  char v12; // [rsp+68h] [rbp-38h]

  v3 = (__int64)a3;
  v8 = (__int64 *)*a3;
  sub_22CDEF0((__int64)a1, a2, *a3, *(_QWORD *)(a3[3] + 40), a3[3]);
  v7 = 3;
  while ( 1 )
  {
    memset(v9, 0, sizeof(v9));
    v10 = 0;
    v4 = *(_QWORD *)(v3 + 24);
    if ( *(_BYTE *)v4 != 86 )
    {
      if ( *(_BYTE *)v4 != 84 )
        goto LABEL_8;
      sub_22CB610(
        (__int64)v11,
        a2,
        v8,
        *(_QWORD *)(*(_QWORD *)(v4 - 8)
                  + 32LL * *(unsigned int *)(v4 + 72)
                  + 8LL * (unsigned int)((v3 - *(_QWORD *)(v4 - 8)) >> 5)),
        *(_QWORD *)(v4 + 40),
        0);
      if ( !BYTE8(v10) )
        goto LABEL_25;
      goto LABEL_14;
    }
    if ( !sub_98EF80(*(unsigned __int8 **)(v4 - 96), *(_QWORD *)(a2 + 240), 0, 0, 0) )
      break;
    if ( (unsigned int)sub_BD2910(v3) == 1 )
    {
      sub_22C9ED0((__int64)v11, a2, (__int64)v8, *(_QWORD *)(v4 - 96), 1u, 0, 0);
      goto LABEL_24;
    }
    if ( (unsigned int)sub_BD2910(v3) == 2 )
    {
      sub_22C9ED0((__int64)v11, a2, (__int64)v8, *(_QWORD *)(v4 - 96), 0, 0, 0);
LABEL_24:
      if ( !BYTE8(v10) )
      {
LABEL_25:
        sub_22C0650((__int64)v9, v11);
        BYTE8(v10) = 1;
LABEL_15:
        if ( v12 )
        {
          v12 = 0;
          sub_22C0090(v11);
        }
        goto LABEL_6;
      }
LABEL_14:
      sub_22C0090((unsigned __int8 *)v9);
      sub_22C0650((__int64)v9, v11);
      goto LABEL_15;
    }
LABEL_6:
    if ( BYTE8(v10) )
    {
      sub_22EACA0(v11, a1, v9);
      sub_22C0090(a1);
      sub_22C0650((__int64)a1, v11);
      sub_22C0090(v11);
    }
LABEL_8:
    v5 = *(_QWORD *)(v4 + 16);
    if ( !v5 || *(_QWORD *)(v5 + 8) || !sub_991A70((unsigned __int8 *)v4, 0, 0, 0, 0, 0, 0) )
      break;
    v3 = *(_QWORD *)(v4 + 16);
    if ( BYTE8(v10) )
    {
      BYTE8(v10) = 0;
      sub_22C0090((unsigned __int8 *)v9);
    }
    if ( !--v7 )
      return a1;
  }
  if ( BYTE8(v10) )
  {
    BYTE8(v10) = 0;
    sub_22C0090((unsigned __int8 *)v9);
  }
  return a1;
}
