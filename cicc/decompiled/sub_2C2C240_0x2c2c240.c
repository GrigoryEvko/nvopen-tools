// Function: sub_2C2C240
// Address: 0x2c2c240
//
__int64 __fastcall sub_2C2C240(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 result; // rax
  char v5; // dl
  __int64 v6; // rax
  _QWORD *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  bool v14; // [rsp+Fh] [rbp-31h]
  __int64 v15[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = sub_2BF04A0(a2);
  if ( v2 )
  {
    if ( *(_BYTE *)(v2 + 8) == 4 && *(_BYTE *)(v2 + 160) == 83 )
    {
      v10 = **(_QWORD **)(v2 + 48);
      if ( v10 )
      {
        **(_QWORD **)(a1 + 8) = v10;
        v11 = *(_QWORD **)a1;
        v12 = sub_2BF04A0(*(_QWORD *)(*(_QWORD *)(v2 + 48) + 8LL));
        if ( v12 )
        {
          if ( *(_BYTE *)(v12 + 8) == 4 && *(_BYTE *)(v12 + 160) == 70 )
          {
            v13 = **(_QWORD **)(v12 + 48);
            if ( v13 )
            {
              *v11 = v13;
              return 1;
            }
          }
        }
      }
    }
  }
  v3 = sub_2BF04A0(a2);
  result = 0;
  if ( v3 )
  {
    v5 = *(_BYTE *)(v3 + 8);
    switch ( v5 )
    {
      case 9:
        if ( **(_BYTE **)(v3 + 136) != 86 )
          return result;
        break;
      case 4:
        if ( *(_BYTE *)(v3 + 160) != 57 )
          return result;
        break;
      case 24:
        break;
      default:
        return result;
    }
    v6 = **(_QWORD **)(v3 + 48);
    if ( v6
      && (**(_QWORD **)(a1 + 40) = v6,
          v7 = *(_QWORD **)(a1 + 32),
          (v8 = sub_2BF04A0(*(_QWORD *)(*(_QWORD *)(v3 + 48) + 8LL))) != 0)
      && *(_BYTE *)(v8 + 8) == 4
      && *(_BYTE *)(v8 + 160) == 70
      && (v9 = **(_QWORD **)(v8 + 48)) != 0 )
    {
      *v7 = v9;
      sub_9865C0((__int64)v15, a1 + 16);
      v14 = sub_2C23EC0((__int64)v15, *(_QWORD *)(*(_QWORD *)(v3 + 48) + 16LL));
      sub_969240(v15);
      return v14;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
