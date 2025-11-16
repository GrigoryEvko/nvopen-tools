// Function: sub_88DE40
// Address: 0x88de40
//
_QWORD *__fastcall sub_88DE40(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 i; // rax
  __int64 **v4; // r14
  _QWORD *v5; // rax
  __int64 v6; // rbx
  __int64 *v7; // r13
  int v8; // r14d
  _QWORD *j; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v14; // [rsp+8h] [rbp-48h]
  __int64 **v15; // [rsp+10h] [rbp-40h]
  _QWORD *v16; // [rsp+18h] [rbp-38h]

  v2 = sub_72C930();
  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4 = *(__int64 ***)(i + 168);
  v15 = v4;
  v5 = sub_7259C0(7);
  v6 = v5[21];
  v14 = v5;
  *(_BYTE *)(v6 + 16) |= 2u;
  v5[20] = v2;
  if ( ((_BYTE)v4[2] & 1) != 0 )
    *(_BYTE *)(v6 + 16) |= 1u;
  v7 = *v4;
  if ( *v4 )
  {
    v8 = 0;
    for ( j = 0; ; j = v11 )
    {
      if ( (*((_BYTE *)v7 + 33) & 1) != 0 )
        *(_BYTE *)(v6 + 20) |= 0x10u;
      ++v8;
      v10 = sub_724EF0(v2);
      *((_DWORD *)v10 + 9) = v8;
      v11 = v10;
      *((_BYTE *)v10 + 33) = (2 * (*((_BYTE *)v7 + 33) & 1)) | *((_BYTE *)v10 + 33) & 0xFD;
      if ( (v7[4] & 4) != 0 )
      {
        *((_BYTE *)v10 + 32) |= 4u;
        v16 = v10;
        v12 = sub_7305E0();
        v11 = v16;
        v16[5] = v12;
        if ( j )
        {
LABEL_8:
          *j = v11;
          v7 = (__int64 *)*v7;
          if ( !v7 )
            break;
          continue;
        }
      }
      else if ( j )
      {
        goto LABEL_8;
      }
      *(_QWORD *)v6 = v11;
      v7 = (__int64 *)*v7;
      if ( !v7 )
        break;
    }
  }
  if ( v15[5] )
  {
    *(_BYTE *)(v6 + 21) |= 1u;
    *(_QWORD *)(v6 + 40) = a2;
  }
  return v14;
}
