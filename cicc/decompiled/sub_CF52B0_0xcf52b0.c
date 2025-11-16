// Function: sub_CF52B0
// Address: 0xcf52b0
//
__int64 __fastcall sub_CF52B0(_QWORD *a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // r15
  int v7; // ebx
  int v9; // eax
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int8 *v17; // rbx
  unsigned int v18; // r13d
  unsigned __int8 *v19; // r12
  unsigned int v20; // edx
  unsigned __int8 v21; // [rsp+Dh] [rbp-83h]
  char v22; // [rsp+Eh] [rbp-82h]
  char v23; // [rsp+Fh] [rbp-81h]
  __int64 v24; // [rsp+10h] [rbp-80h]
  __int64 v25; // [rsp+18h] [rbp-78h]
  int v26; // [rsp+18h] [rbp-78h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  _QWORD *v28; // [rsp+28h] [rbp-68h]
  unsigned __int8 v29; // [rsp+28h] [rbp-68h]
  _BYTE v30[96]; // [rsp+30h] [rbp-60h] BYREF

  v28 = (_QWORD *)a1[2];
  if ( (_QWORD *)a1[1] != v28 )
  {
    v6 = (_QWORD *)a1[1];
    v7 = 3;
    while ( 1 )
    {
      LOBYTE(v7) = (*(__int64 (__fastcall **)(_QWORD, unsigned __int8 *, __int64, __int64))(*(_QWORD *)*v6 + 56LL))(
                     *v6,
                     a2,
                     a3,
                     a4)
                 & v7;
      if ( !(_BYTE)v7 )
        return 0;
      if ( v28 == ++v6 )
        goto LABEL_7;
    }
  }
  v7 = 3;
LABEL_7:
  v9 = sub_CF5230((__int64)a1, (__int64)a2, a4);
  if ( (v9 & 0xFFFFFFF3) == 0 )
    return 0;
  v23 = v9 & 3;
  v29 = ((v9 & 0xFFFFFFF0) >> 4) & 3 | ((v9 & 0xFFFFFFF0) >> 2) & 3 | ((v9 & 0xFFFFFFF0) >> 6) & 3;
  if ( (v9 & 3 | v29) == v29 )
    goto LABEL_9;
  v10 = *a2;
  if ( v10 == 40 )
  {
    v11 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v11 = -32;
    if ( v10 != 85 )
    {
      v11 = -96;
      if ( v10 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v12 = sub_BD2BC0((__int64)a2);
    v25 = v13 + v12;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v25 >> 4) )
        goto LABEL_22;
    }
    else
    {
      if ( !(unsigned int)((v25 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_22;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v26 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v14 = sub_BD2BC0((__int64)a2);
        v11 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v26);
        goto LABEL_22;
      }
    }
    BUG();
  }
LABEL_22:
  v16 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( &a2[-v16] != &a2[v11] )
  {
    v21 = v7;
    v17 = &a2[v11];
    v27 = a3;
    v18 = 0;
    v24 = a4;
    v19 = &a2[-v16];
    v22 = 0;
    do
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v19 + 8LL) + 8LL) == 14 )
        {
          sub_D669C0(v30, a2, v18, *a1);
          if ( (unsigned __int8)sub_CF4D50((__int64)a1, (__int64)v30, v27, v24, (__int64)a2) )
            break;
        }
        v19 += 32;
        ++v18;
        if ( v17 == v19 )
          goto LABEL_28;
      }
      v20 = v18;
      v19 += 32;
      ++v18;
      v22 |= sub_CF51C0((__int64)a1, (__int64)a2, v20);
    }
    while ( v17 != v19 );
LABEL_28:
    v7 = v21;
    a3 = v27;
    LOBYTE(v7) = (v22 & v23 | v29) & v21;
    if ( (_BYTE)v7 )
      return v7 & (unsigned int)sub_CF5020((__int64)a1, a3, 0);
    return 0;
  }
LABEL_9:
  LOBYTE(v7) = v29 & v7;
  if ( !(_BYTE)v7 )
    return 0;
  return v7 & (unsigned int)sub_CF5020((__int64)a1, a3, 0);
}
