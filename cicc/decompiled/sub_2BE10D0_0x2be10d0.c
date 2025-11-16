// Function: sub_2BE10D0
// Address: 0x2be10d0
//
__int64 __fastcall sub_2BE10D0(_QWORD *a1, __int64 a2, char *a3, char a4)
{
  __int64 v4; // rax
  char *v5; // r12
  __int64 v6; // r15
  __int64 (__fastcall *v7)(__int64, unsigned int); // r8
  unsigned __int64 v8; // rdx
  size_t v9; // rax
  unsigned __int64 v10; // rcx
  size_t v11; // r11
  char v12; // al
  __int64 v13; // rcx
  char v14; // bl
  _UNKNOWN **v15; // rbx
  const char *i; // rsi
  unsigned int v17; // r12d
  size_t v20; // [rsp+18h] [rbp-68h]
  size_t v21; // [rsp+20h] [rbp-60h]
  __int64 v22; // [rsp+20h] [rbp-60h]
  _QWORD *v24; // [rsp+30h] [rbp-50h] BYREF
  size_t v25; // [rsp+38h] [rbp-48h]
  _QWORD v26[8]; // [rsp+40h] [rbp-40h] BYREF

  v4 = sub_222F790(a1, a2);
  v24 = v26;
  v25 = 0;
  LOBYTE(v26[0]) = 0;
  if ( (char *)a2 != a3 )
  {
    v5 = (char *)a2;
    v6 = v4;
    do
    {
      v13 = (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v6 + 32LL))(v6, (unsigned int)*v5);
      v14 = v13;
      if ( *(_BYTE *)(v6 + v13 + 313) )
      {
        v14 = *(_BYTE *)(v6 + v13 + 313);
      }
      else
      {
        v7 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v6 + 64LL);
        if ( v7 != sub_2216C50 )
        {
          v22 = v13;
          v12 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v7)(v6, (unsigned int)(char)v13, 0);
          v13 = v22;
          v14 = v12;
        }
        if ( v14 )
          *(_BYTE *)(v6 + v13 + 313) = v14;
      }
      v8 = (unsigned __int64)v24;
      v9 = v25;
      v10 = 15;
      if ( v24 != v26 )
        v10 = v26[0];
      v11 = v25 + 1;
      if ( v25 + 1 > v10 )
      {
        v20 = v25 + 1;
        v21 = v25;
        sub_2240BB0((unsigned __int64 *)&v24, v25, 0, 0, 1u);
        v8 = (unsigned __int64)v24;
        v11 = v20;
        v9 = v21;
      }
      *(_BYTE *)(v8 + v9) = v14;
      ++v5;
      v25 = v11;
      *((_BYTE *)v24 + v9 + 1) = 0;
    }
    while ( a3 != v5 );
  }
  v15 = &off_49D3EA0;
  for ( i = "d"; sub_2241AC0((__int64)&v24, i); i = (const char *)*v15 )
  {
    v15 += 2;
    if ( &unk_49D3F90 == (_UNKNOWN *)v15 )
    {
      v17 = 0;
      goto LABEL_19;
    }
  }
  if ( !a4 || (v17 = 1024, ((_WORD)v15[1] & 0x300) == 0) )
    v17 = *((_DWORD *)v15 + 2);
LABEL_19:
  if ( v24 != v26 )
    j_j___libc_free_0((unsigned __int64)v24);
  return v17;
}
