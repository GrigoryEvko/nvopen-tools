// Function: sub_AC4540
// Address: 0xac4540
//
__int64 __fastcall sub_AC4540(__int64 a1, __int64 a2, char *a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned int v8; // eax
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  char *v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // [rsp+8h] [rbp-58h]
  __int64 v16; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-38h]

  if ( *(_QWORD *)(a1 - 96) != a2 )
    return 0;
  LOBYTE(v8) = sub_AC30F0(*(_QWORD *)(a1 - 32));
  v4 = v8;
  if ( (_BYTE)v8 )
  {
    LOBYTE(v4) = *(_QWORD *)(a1 - 64) == (_QWORD)a3;
    return v4;
  }
  if ( !sub_AC30F0(*(_QWORD *)(a1 - 64)) )
  {
    if ( *a3 != 85 )
      return 0;
    v12 = *((_QWORD *)a3 - 4);
    if ( !v12 )
      return 0;
    if ( *(_BYTE *)v12 )
      return 0;
    if ( *(_QWORD *)(v12 + 24) != *((_QWORD *)a3 + 10) )
      return 0;
    if ( *(_DWORD *)(v12 + 36) != 294 )
      return 0;
    v13 = a3;
    v14 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
    a3 = *(char **)&a3[-32 * v14];
    if ( !a3 || *(_QWORD *)(a1 - 64) != *(_QWORD *)&v13[32 * (1 - v14)] )
      return 0;
  }
  v9 = *a3;
  if ( (unsigned __int8)*a3 > 0x1Cu )
  {
    if ( v9 != 76 )
      goto LABEL_10;
    goto LABEL_24;
  }
  if ( v9 == 5 && *((_WORD *)a3 + 1) == 47 )
LABEL_24:
    a3 = (char *)*((_QWORD *)a3 - 4);
LABEL_10:
  v10 = *(_QWORD *)(a1 - 32);
  v11 = *(_QWORD *)(v10 + 8);
  if ( v11 != *((_QWORD *)a3 + 1) )
    return 0;
  if ( (char *)v10 == a3 )
  {
    return 1;
  }
  else
  {
    v17 = sub_AE43F0(a4, v11);
    if ( v17 > 0x40 )
      sub_C43690(&v16, 0, 0);
    else
      v16 = 0;
    v15 = sub_BD45C0(*(_QWORD *)(a1 - 32), a4, (unsigned int)&v16, 1, 0, 0, 0, 0);
    v19 = sub_AE43F0(a4, *((_QWORD *)a3 + 1));
    if ( v19 > 0x40 )
      sub_C43690(&v18, 0, 0);
    else
      v18 = 0;
    if ( v15 == sub_BD45C0((_DWORD)a3, a4, (unsigned int)&v18, 1, 0, 0, 0, 0) )
    {
      if ( v17 <= 0x40 )
        LOBYTE(v4) = v16 == v18;
      else
        v4 = sub_C43C50(&v16, &v18);
    }
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
  }
  return v4;
}
