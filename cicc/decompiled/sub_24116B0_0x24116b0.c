// Function: sub_24116B0
// Address: 0x24116b0
//
unsigned __int8 *__fastcall sub_24116B0(
        unsigned __int8 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int8 *v6; // r12
  __int64 v8; // rdx
  int i; // eax
  int v10; // eax
  unsigned __int8 **v11; // rax
  __int64 v12; // rdx
  unsigned __int8 *v13; // r12
  __int64 v14; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int8 **v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h]
  int v17; // [rsp+18h] [rbp-38h]
  unsigned __int8 v18; // [rsp+1Ch] [rbp-34h]
  unsigned __int8 *v19; // [rsp+20h] [rbp-30h] BYREF

  v6 = a1;
  if ( *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) != 14 )
    return v6;
  v17 = 0;
  v8 = 1;
  v15 = &v19;
  v16 = 0x100000004LL;
  v18 = 1;
  v19 = a1;
  v14 = 1;
LABEL_4:
  for ( i = *v6; (unsigned __int8)i <= 0x1Cu; i = *v6 )
  {
    if ( (_BYTE)i == 5 )
    {
      v10 = *((unsigned __int16 *)v6 + 1);
      if ( (_WORD)v10 != 34 )
        goto LABEL_7;
LABEL_16:
      v6 = *(unsigned __int8 **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
      if ( !(_BYTE)v8 )
      {
LABEL_17:
        sub_C8CC70((__int64)&v14, (__int64)v6, v8, a4, a5, a6);
        a5 = v12;
        v8 = v18;
        if ( !(_BYTE)a5 )
          goto LABEL_18;
        goto LABEL_4;
      }
    }
    else
    {
      if ( (_BYTE)i == 1 )
        v6 = (unsigned __int8 *)*((_QWORD *)v6 - 4);
LABEL_8:
      if ( !(_BYTE)v8 )
        goto LABEL_17;
    }
    v11 = v15;
    a4 = HIDWORD(v16);
    v8 = (__int64)&v15[HIDWORD(v16)];
    if ( v15 != (unsigned __int8 **)v8 )
    {
      while ( v6 != *v11 )
      {
        if ( (unsigned __int8 **)v8 == ++v11 )
          goto LABEL_12;
      }
      return v6;
    }
LABEL_12:
    if ( HIDWORD(v16) >= (unsigned int)v16 )
      goto LABEL_17;
    a4 = (unsigned int)++HIDWORD(v16);
    *(_QWORD *)v8 = v6;
    v8 = v18;
    ++v14;
  }
  if ( (_BYTE)i == 63 )
    goto LABEL_16;
  v10 = i - 29;
LABEL_7:
  if ( v10 != 49 )
    goto LABEL_8;
  v13 = (v6[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v6 - 1) : &v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
  v6 = *(unsigned __int8 **)v13;
  if ( *(_BYTE *)(*((_QWORD *)v6 + 1) + 8LL) == 14 )
    goto LABEL_8;
LABEL_18:
  if ( (_BYTE)v8 )
    return v6;
  _libc_free((unsigned __int64)v15);
  return v6;
}
