// Function: sub_36D4AB0
// Address: 0x36d4ab0
//
__int64 __fastcall sub_36D4AB0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rbx
  _BYTE *v6; // r14
  __int64 v7; // r11
  _QWORD *v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 result; // rax
  const void *i; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+10h] [rbp-40h]
  int v18; // [rsp+18h] [rbp-38h]

  v5 = *(_QWORD **)(a2 + 16);
  v16 = a1 + 176;
  for ( i = (const void *)(a1 + 192); v5; v5 = (_QWORD *)v5[1] )
  {
    while ( 1 )
    {
      v6 = (_BYTE *)*v5;
      if ( *(_BYTE *)*v5 == 31 && (*((_DWORD *)v6 + 1) & 0x7FFFFFF) != 1 )
        break;
      v5 = (_QWORD *)v5[1];
      if ( !v5 )
        goto LABEL_14;
    }
    if ( *(_DWORD *)(a3 + 32) <= 0x40u )
    {
      if ( !*(_QWORD *)(a3 + 24) )
      {
LABEL_18:
        v7 = *((_QWORD *)v6 - 8);
        goto LABEL_9;
      }
    }
    else
    {
      v18 = *(_DWORD *)(a3 + 32);
      if ( v18 == (unsigned int)sub_C444A0(a3 + 24) )
        goto LABEL_18;
    }
    v7 = *((_QWORD *)v6 - 4);
LABEL_9:
    v17 = v7;
    v8 = sub_BD2C40(72, 1u);
    if ( v8 )
      sub_B4C8F0((__int64)v8, v17, 1u, (__int64)(v6 + 24), 0);
    v11 = *(unsigned int *)(a1 + 184);
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 188) )
    {
      sub_C8D5F0(v16, i, v11 + 1, 8u, v9, v10);
      v11 = *(unsigned int *)(a1 + 184);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8 * v11) = v6;
    ++*(_DWORD *)(a1 + 184);
  }
LABEL_14:
  sub_BD84D0(a2, a3);
  result = *(unsigned int *)(a1 + 184);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 188) )
  {
    sub_C8D5F0(v16, (const void *)(a1 + 192), result + 1, 8u, v12, v13);
    result = *(unsigned int *)(a1 + 184);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 184);
  return result;
}
