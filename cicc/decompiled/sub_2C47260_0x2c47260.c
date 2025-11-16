// Function: sub_2C47260
// Address: 0x2c47260
//
__int64 __fastcall sub_2C47260(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  unsigned int v6; // r13d
  _BYTE *v7; // r12
  unsigned int v8; // edx
  bool v9; // bl
  __int64 v10; // rdx
  _BYTE *v11; // rax
  unsigned __int8 v12; // [rsp+Fh] [rbp-41h]
  const void *v13; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-38h]
  const void *v15; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-28h]

  v2 = sub_2BF04A0(a2);
  result = 0;
  if ( v2 && *(_BYTE *)(v2 + 8) == 11 )
  {
    v4 = sub_2BF04A0(**(_QWORD **)(v2 + 48));
    if ( !v4 || *(_BYTE *)(v4 + 8) != 29 )
      return 0;
    v14 = *(_DWORD *)(a1 + 8);
    if ( v14 > 0x40 )
      sub_C43780((__int64)&v13, (const void **)a1);
    else
      v13 = *(const void **)a1;
    v5 = *(_QWORD *)(*(_QWORD *)(v2 + 48) + 8LL);
    if ( sub_2BF04A0(v5)
      || (v7 = *(_BYTE **)(v5 + 40)) == 0
      || *v7 != 17
      && ((v10 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v7 + 1) + 8LL) - 17, (unsigned int)v10 > 1)
       || *v7 > 0x15u
       || (v11 = sub_AD7630(*(_QWORD *)(v5 + 40), 0, v10), (v7 = v11) == 0)
       || *v11 != 17) )
    {
      v6 = v14;
      goto LABEL_9;
    }
    v8 = *((_DWORD *)v7 + 8);
    v6 = v14;
    if ( v8 == v14 )
    {
      if ( v14 <= 0x40 )
      {
        if ( *((const void **)v7 + 3) == v13 )
          goto LABEL_25;
        goto LABEL_9;
      }
      v9 = sub_C43C50((__int64)(v7 + 24), &v13);
LABEL_24:
      if ( v9 )
      {
LABEL_25:
        result = 1;
        if ( v6 <= 0x40 )
          return result;
        result = 1;
        goto LABEL_10;
      }
LABEL_9:
      result = 0;
      if ( v6 <= 0x40 )
        return result;
LABEL_10:
      if ( v13 )
      {
        v12 = result;
        j_j___libc_free_0_0((unsigned __int64)v13);
        return v12;
      }
      return result;
    }
    if ( v8 <= v14 )
    {
      sub_C449B0((__int64)&v15, (const void **)v7 + 3, v14);
      if ( v16 <= 0x40 )
      {
        v9 = v15 == v13;
LABEL_23:
        v6 = v14;
        goto LABEL_24;
      }
      v9 = sub_C43C50((__int64)&v15, &v13);
    }
    else
    {
      sub_C449B0((__int64)&v15, &v13, v8);
      if ( *((_DWORD *)v7 + 8) <= 0x40u )
        v9 = *((_QWORD *)v7 + 3) == (_QWORD)v15;
      else
        v9 = sub_C43C50((__int64)(v7 + 24), &v15);
      if ( v16 <= 0x40 )
        goto LABEL_23;
    }
    if ( v15 )
      j_j___libc_free_0_0((unsigned __int64)v15);
    goto LABEL_23;
  }
  return result;
}
