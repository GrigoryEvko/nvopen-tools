// Function: sub_2A65320
// Address: 0x2a65320
//
__int64 __fastcall sub_2A65320(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // eax
  unsigned int v5; // r12d
  _BYTE *v7; // rax
  char v8; // r13
  _BYTE *v9; // rbx
  unsigned int v10; // eax
  unsigned int v11; // esi
  unsigned int v12; // eax
  const void **v13; // r12
  unsigned __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-38h]
  unsigned __int64 v16; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-28h]

  if ( *(_BYTE *)a2 <= 0x15u )
  {
    sub_AD8380((__int64)&v14, a2);
    goto LABEL_3;
  }
  if ( (unsigned __int8)sub_B19060(a1[1], a2, a3, a4) )
  {
    v12 = sub_BCB060(*(_QWORD *)(a2 + 8));
    sub_AADB10((__int64)&v14, v12, 1);
    goto LABEL_3;
  }
  v7 = (_BYTE *)sub_2A64F10(*a1, a2);
  v8 = *v7;
  v9 = v7;
  if ( *v7 == 4 )
  {
    v13 = (const void **)(v7 + 8);
  }
  else
  {
    v10 = sub_BCB060(*(_QWORD *)(a2 + 8));
    v11 = v10;
    if ( v8 != 5 )
      goto LABEL_13;
    v13 = (const void **)(v9 + 8);
    v11 = v10;
    if ( !sub_9876C0((__int64 *)v9 + 1) )
    {
      v8 = *v9;
LABEL_13:
      if ( v8 == 2 )
      {
        sub_AD8380((__int64)&v14, *((_QWORD *)v9 + 1));
      }
      else if ( v8 )
      {
        sub_AADB10((__int64)&v14, v11, 1);
      }
      else
      {
        sub_AADB10((__int64)&v14, v11, 0);
      }
      goto LABEL_3;
    }
  }
  v15 = *((_DWORD *)v9 + 4);
  if ( v15 > 0x40 )
    sub_C43780((__int64)&v14, v13);
  else
    v14 = *((_QWORD *)v9 + 1);
  v17 = *((_DWORD *)v9 + 8);
  if ( v17 > 0x40 )
    sub_C43780((__int64)&v16, (const void **)v9 + 3);
  else
    v16 = *((_QWORD *)v9 + 3);
LABEL_3:
  LOBYTE(v4) = sub_AB0760((__int64)&v14);
  v5 = v4;
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return v5;
}
