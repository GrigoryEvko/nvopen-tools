// Function: sub_2C2C0F0
// Address: 0x2c2c0f0
//
__int64 __fastcall sub_2C2C0F0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 result; // rax
  char v5; // dl
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  bool v10; // [rsp+Fh] [rbp-31h]
  __int64 v11[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = sub_2BF04A0(a2);
  if ( v2 )
  {
    if ( *(_BYTE *)(v2 + 8) == 4 && *(_BYTE *)(v2 + 160) == 83 )
    {
      v8 = **(_QWORD **)(v2 + 48);
      if ( v8 )
      {
        *a1[1] = v8;
        v9 = *(_QWORD *)(*(_QWORD *)(v2 + 48) + 8LL);
        if ( v9 )
        {
          **a1 = v9;
          return 1;
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
    if ( v6 && (*a1[5] = v6, (v7 = *(_QWORD *)(*(_QWORD *)(v3 + 48) + 8LL)) != 0) )
    {
      *a1[4] = v7;
      sub_9865C0((__int64)v11, (__int64)(a1 + 2));
      v10 = sub_2C23EC0((__int64)v11, *(_QWORD *)(*(_QWORD *)(v3 + 48) + 16LL));
      sub_969240(v11);
      return v10;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
