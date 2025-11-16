// Function: sub_BD4CB0
// Address: 0xbd4cb0
//
unsigned __int8 *__fastcall sub_BD4CB0(
        unsigned __int8 *a1,
        void (__fastcall *a2)(__int64, unsigned __int8 *),
        __int64 a3)
{
  unsigned __int8 *v3; // r12
  __int64 v7; // r14
  __int64 v8; // rsi
  int v9; // eax
  unsigned __int8 *v10; // r12
  char v11; // al
  char v12; // dl
  unsigned __int8 **v13; // rax
  unsigned __int8 **v14; // rdx
  __int16 v15; // ax
  __int64 *v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int8 **v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  int v22; // [rsp+18h] [rbp-48h]
  char v23; // [rsp+1Ch] [rbp-44h]
  unsigned __int8 *v24; // [rsp+20h] [rbp-40h] BYREF

  v3 = a1;
  if ( *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) != 14 )
    return v3;
  v23 = 1;
  v20 = &v24;
  v7 = 0x8000000000041LL;
  v21 = 0x100000004LL;
  v22 = 0;
  v24 = a1;
  v19 = 1;
  while ( 1 )
  {
    while ( 1 )
    {
      v8 = (__int64)v3;
      a2(a3, v3);
      v9 = *v3;
      if ( (unsigned __int8)v9 > 0x1Cu )
        break;
      if ( (_BYTE)v9 != 5 )
        goto LABEL_31;
      v15 = *((_WORD *)v3 + 1);
      if ( v15 == 34 )
        goto LABEL_13;
      if ( v15 == 49 )
        goto LABEL_24;
      if ( v15 != 50 )
        goto LABEL_31;
LABEL_8:
      if ( (v3[7] & 0x40) != 0 )
        v10 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
      else
        v10 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
      v3 = *(unsigned __int8 **)v10;
      if ( !v23 )
        goto LABEL_11;
LABEL_16:
      v13 = v20;
      v14 = &v20[HIDWORD(v21)];
      if ( v20 != v14 )
      {
        while ( v3 != *v13 )
        {
          if ( v14 == ++v13 )
            goto LABEL_19;
        }
        return v3;
      }
LABEL_19:
      if ( HIDWORD(v21) >= (unsigned int)v21 )
        goto LABEL_11;
      ++HIDWORD(v21);
      *v14 = v3;
      ++v19;
    }
    if ( (_BYTE)v9 != 63 )
      break;
LABEL_13:
    if ( (v3[1] & 2) == 0 )
      goto LABEL_31;
    v3 = *(unsigned __int8 **)&v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
LABEL_15:
    if ( v23 )
      goto LABEL_16;
LABEL_11:
    v8 = (__int64)v3;
    sub_C8CC70(&v19, v3);
    v11 = v23;
    if ( !v12 )
      goto LABEL_32;
  }
  if ( (unsigned __int8)v9 != 78 )
  {
    if ( (unsigned __int8)v9 == 79 )
      goto LABEL_8;
    v18 = (unsigned int)(v9 - 34);
    if ( (unsigned __int8)v18 > 0x33u )
      goto LABEL_31;
    if ( !_bittest64(&v7, v18) )
      goto LABEL_31;
    v8 = 52;
    v17 = sub_B494D0((__int64)v3, 52);
    if ( !v17 )
      goto LABEL_31;
    goto LABEL_27;
  }
LABEL_24:
  if ( (v3[7] & 0x40) != 0 )
    v16 = (__int64 *)*((_QWORD *)v3 - 1);
  else
    v16 = (__int64 *)&v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
  v17 = *v16;
  if ( *(_BYTE *)(*(_QWORD *)(*v16 + 8) + 8LL) == 14 )
  {
LABEL_27:
    v3 = (unsigned __int8 *)v17;
    goto LABEL_15;
  }
LABEL_31:
  v11 = v23;
LABEL_32:
  if ( v11 )
    return v3;
  _libc_free(v20, v8);
  return v3;
}
