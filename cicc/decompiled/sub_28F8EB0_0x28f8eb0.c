// Function: sub_28F8EB0
// Address: 0x28f8eb0
//
unsigned __int8 *__fastcall sub_28F8EB0(__int64 a1, unsigned __int8 *a2, _DWORD *a3)
{
  __int64 v5; // rax
  int v6; // ebx
  unsigned int v7; // edx
  __int64 v8; // rcx
  unsigned int v9; // r13d
  unsigned __int8 *v10; // r15
  __int64 v11; // rax
  _BYTE *v12; // rsi
  __int64 v13; // rax
  int v15; // ecx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned __int64 v19; // r11
  unsigned __int64 *v20; // rax
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+18h] [rbp-38h]
  int v23; // [rsp+18h] [rbp-38h]
  unsigned __int64 v24; // [rsp+18h] [rbp-38h]

  while ( 2 )
  {
    v5 = sub_B43CC0((__int64)a2);
    v6 = *a2;
    v7 = a3[2];
    v8 = v5;
    v9 = v6 - 29;
    if ( !v7 )
      return 0;
    v10 = 0;
    while ( 1 )
    {
      v11 = *(_QWORD *)a3 + 16LL * v7;
      v12 = *(_BYTE **)(v11 - 8);
      if ( *v12 > 0x15u )
      {
        v15 = a3[2];
        if ( !v15 )
          return v10;
        if ( v10 )
          goto LABEL_25;
        goto LABEL_11;
      }
      if ( !v10 )
      {
        --v7;
        v10 = *(unsigned __int8 **)(v11 - 8);
        a3[2] = v7;
        goto LABEL_7;
      }
      v22 = v8;
      v13 = sub_96E6C0(v9, (__int64)v12, v10, v8);
      if ( !v13 )
        break;
      v8 = v22;
      v10 = (unsigned __int8 *)v13;
      v7 = a3[2] - 1;
      a3[2] = v7;
LABEL_7:
      if ( !v7 )
        return v10;
    }
    if ( !a3[2] )
      return v10;
LABEL_25:
    if ( v10 != sub_AD93D0(v9, *((_QWORD *)a2 + 1), 0, 0) )
    {
      if ( v10 == (unsigned __int8 *)sub_AD6840(v9, *((_QWORD *)a2 + 1), 0) )
        return v10;
      v18 = (unsigned int)a3[2];
      v19 = v21 & 0xFFFFFFFF00000000LL;
      v21 &= 0xFFFFFFFF00000000LL;
      if ( v18 + 1 > (unsigned __int64)(unsigned int)a3[3] )
      {
        v24 = v19;
        sub_C8D5F0((__int64)a3, a3 + 4, v18 + 1, 0x10u, v16, v17);
        v18 = (unsigned int)a3[2];
        v19 = v24;
      }
      v20 = (unsigned __int64 *)(*(_QWORD *)a3 + 16 * v18);
      *v20 = v19;
      v20[1] = (unsigned __int64)v10;
      LODWORD(v20) = a3[2];
      v15 = (_DWORD)v20 + 1;
      a3[2] = (_DWORD)v20 + 1;
      if ( !(_DWORD)v20 )
        return *(unsigned __int8 **)(*(_QWORD *)a3 + 8LL);
      goto LABEL_12;
    }
    v15 = a3[2];
LABEL_11:
    if ( v15 == 1 )
      return *(unsigned __int8 **)(*(_QWORD *)a3 + 8LL);
LABEL_12:
    if ( v9 > 0x1D )
    {
      if ( v6 != 59 )
        return 0;
      v23 = v15;
      v10 = (unsigned __int8 *)sub_28F6270(a1, (__int64)a2, (__int64)a3);
      if ( !v10 )
        goto LABEL_19;
    }
    else
    {
      if ( v9 > 0x1B )
      {
        v23 = v15;
        v10 = (unsigned __int8 *)sub_28EB620(v9, (__int64)a3);
        if ( v10 )
          return v10;
        goto LABEL_19;
      }
      if ( v9 <= 0xE )
      {
        if ( v9 <= 0xC )
          return 0;
        v23 = v15;
        v10 = (unsigned __int8 *)sub_28F7360(a1, (__int64)a2, (__int64)a3);
        if ( v10 )
          return v10;
LABEL_19:
        if ( v23 != a3[2] )
          continue;
        return 0;
      }
      if ( (unsigned int)(v6 - 46) > 1 )
        return 0;
      v23 = v15;
      v10 = sub_28F4730(a1, (__int64)a2, (__int64)a3);
      if ( !v10 )
        goto LABEL_19;
    }
    return v10;
  }
}
