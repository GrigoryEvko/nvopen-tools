// Function: sub_2C1BCC0
// Address: 0x2c1bcc0
//
bool __fastcall sub_2C1BCC0(__int64 a1, __int64 a2)
{
  char v3; // al
  bool result; // al
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdx
  _BYTE *v10; // rax
  bool v11; // [rsp+Fh] [rbp-31h]
  unsigned __int64 v12; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-28h]

  v3 = *(_BYTE *)(a2 + 8);
  switch ( v3 )
  {
    case 9:
      if ( **(_BYTE **)(a2 + 136) != 86 )
        return 0;
      break;
    case 4:
      if ( *(_BYTE *)(a2 + 160) != 57 )
        return 0;
      v5 = **(_QWORD **)(a2 + 48);
      result = 0;
      if ( !v5 )
        return result;
      goto LABEL_7;
    case 24:
      break;
    default:
      return 0;
  }
  v5 = **(_QWORD **)(a2 + 48);
  result = 0;
  if ( !v5 )
    return result;
LABEL_7:
  **(_QWORD **)(a1 + 24) = v5;
  v13 = *(_DWORD *)(a1 + 16);
  if ( v13 > 0x40 )
    sub_C43780((__int64)&v12, (const void **)(a1 + 8));
  else
    v12 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  if ( !sub_2BF04A0(v6)
    && (v7 = *(_QWORD *)(v6 + 40)) != 0
    && (*(_BYTE *)v7 == 17
     || (v9 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17, (unsigned int)v9 <= 1)
     && *(_BYTE *)v7 <= 0x15u
     && (v10 = sub_AD7630(v7, 0, v9), (v7 = (__int64)v10) != 0)
     && *v10 == 17)
    && *(_DWORD *)(v7 + 32) == 1
    && (result = sub_1112D90(v7 + 24, (__int64)&v12))
    && (v8 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL)) != 0 )
  {
    **(_QWORD **)a1 = v8;
    if ( v13 <= 0x40 )
      return result;
  }
  else
  {
    result = 0;
    if ( v13 <= 0x40 )
      return result;
  }
  if ( v12 )
  {
    v11 = result;
    j_j___libc_free_0_0(v12);
    return v11;
  }
  return result;
}
