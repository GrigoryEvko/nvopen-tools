// Function: sub_2C2C7C0
// Address: 0x2c2c7c0
//
__int64 __fastcall sub_2C2C7C0(__int64 a1, __int64 a2)
{
  char v2; // al
  _QWORD *v4; // rax
  _QWORD *v5; // rcx
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(_BYTE *)(a2 + 8);
  switch ( v2 )
  {
    case 23:
      goto LABEL_2;
    case 9:
      if ( **(_BYTE **)(a2 + 136) != 46 )
        return 0;
      goto LABEL_6;
    case 16:
LABEL_2:
      if ( *(_DWORD *)(a2 + 160) != 17 )
        return 0;
LABEL_6:
      v4 = *(_QWORD **)(a2 + 48);
      v5 = *(_QWORD **)(a1 + 16);
      if ( *v4 )
      {
        *v5 = *v4;
        sub_9865C0((__int64)v10, a1);
        LOBYTE(v6) = sub_2C2C640((__int64)v10, *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
        v7 = v6;
        if ( (_BYTE)v6 )
        {
          sub_969240(v10);
          return v7;
        }
        sub_969240(v10);
        v5 = *(_QWORD **)(a1 + 16);
        v4 = *(_QWORD **)(a2 + 48);
      }
      v8 = v4[*(_DWORD *)(a2 + 56) - 1];
      if ( !v8 )
        return 0;
      *v5 = v8;
      sub_9865C0((__int64)v10, a1);
      LOBYTE(v9) = sub_2C2C640(
                     (__int64)v10,
                     *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a2 + 56) - 2)));
      v7 = v9;
      sub_969240(v10);
      return v7;
  }
  if ( v2 != 4 )
    return 0;
  if ( *(_BYTE *)(a2 + 160) == 17 )
    goto LABEL_6;
  return 0;
}
