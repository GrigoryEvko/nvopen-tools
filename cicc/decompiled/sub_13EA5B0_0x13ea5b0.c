// Function: sub_13EA5B0
// Address: 0x13ea5b0
//
int *__fastcall sub_13EA5B0(int *a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 *v9; // rax
  __int64 v10; // r8
  int v11; // edi
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int8 *v15; // rsi
  unsigned __int8 *v16; // rdx
  __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h]
  __int64 v22; // [rsp+28h] [rbp-38h]
  __int64 v23; // [rsp+30h] [rbp-30h]

  v9 = (__int64 *)sub_15A3C50(*(_QWORD *)a3, a4);
  v11 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v11 <= 0x17u )
    goto LABEL_17;
  if ( (unsigned int)(v11 - 60) > 0xC )
  {
    if ( (unsigned int)(v11 - 35) > 0x11 )
      goto LABEL_17;
    v15 = *(unsigned __int8 **)(a2 - 48);
    v16 = *(unsigned __int8 **)(a2 - 24);
    v19 = a5;
    v20 = 0;
    v21 = 0;
    if ( a3 == v15 )
      v15 = (unsigned __int8 *)v9;
    v22 = 0;
    if ( a3 == v16 )
      v16 = (unsigned __int8 *)v9;
    v23 = 0;
    v13 = (__int64)sub_13E1140(v11 - 24, v15, v16, &v19);
    if ( !v13 )
      goto LABEL_17;
  }
  else
  {
    v19 = a5;
    v12 = *(_QWORD *)a2;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v13 = sub_13D1870((unsigned int)(v11 - 24), v9, v12, &v19, v10);
    if ( !v13 )
    {
LABEL_17:
      *a1 = 4;
      return a1;
    }
  }
  if ( *(_BYTE *)(v13 + 16) != 13 )
    goto LABEL_17;
  v18 = *(_DWORD *)(v13 + 32);
  if ( v18 > 0x40 )
    sub_16A4FD0(&v17, v13 + 24);
  else
    v17 = *(_QWORD *)(v13 + 24);
  sub_1589870(&v19, &v17);
  sub_13EA060(a1, &v19);
  if ( (unsigned int)v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( (unsigned int)v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  return a1;
}
