// Function: sub_BD42C0
// Address: 0xbd42c0
//
unsigned __int8 *__fastcall sub_BD42C0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  __int64 v4; // r13
  int v5; // eax
  unsigned __int8 *v6; // r12
  unsigned __int8 **v7; // rax
  unsigned __int8 **v8; // rdx
  __int16 v9; // ax
  __int64 *v10; // rdx
  __int64 v11; // rax
  char v12; // al
  char v13; // dl
  __int64 v14; // r14
  unsigned __int8 *v15; // rbx
  _BYTE *v16; // rdi
  unsigned int v17; // r15d
  unsigned __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int8 **v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+10h] [rbp-60h]
  int v22; // [rsp+18h] [rbp-58h]
  char v23; // [rsp+1Ch] [rbp-54h]
  unsigned __int8 *v24; // [rsp+20h] [rbp-50h] BYREF

  v2 = a1;
  if ( *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) != 14 )
    return v2;
  v22 = 0;
  v4 = 0x8000000000041LL;
  v20 = &v24;
  v21 = 0x100000004LL;
  v23 = 1;
  v24 = a1;
  v19 = 1;
LABEL_4:
  v5 = *v2;
  if ( (unsigned __int8)v5 > 0x1Cu )
  {
LABEL_5:
    if ( (_BYTE)v5 != 63 )
    {
      if ( (unsigned __int8)v5 != 78 )
      {
        if ( (unsigned __int8)v5 == 79 )
          goto LABEL_8;
        if ( (_BYTE)v5 == 84 )
        {
          if ( (*((_DWORD *)v2 + 1) & 0x7FFFFFF) != 1 )
            goto LABEL_30;
          v2 = (unsigned __int8 *)**((_QWORD **)v2 - 1);
          goto LABEL_11;
        }
        v18 = (unsigned int)(v5 - 34);
        if ( (unsigned __int8)v18 > 0x33u || !_bittest64(&v4, v18) )
          goto LABEL_30;
        a2 = 52;
        v11 = sub_B494D0((__int64)v2, 52);
        if ( !v11 )
        {
          if ( (unsigned int)sub_B49240((__int64)v2) != 208
            && (unsigned int)sub_B49240((__int64)v2) != 346
            && (unsigned int)sub_B49240((__int64)v2) != 8170
            && (unsigned int)sub_B49240((__int64)v2) != 9250
            && (unsigned int)sub_B49240((__int64)v2) != 8923 )
          {
            goto LABEL_30;
          }
          v2 = *(unsigned __int8 **)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
          goto LABEL_11;
        }
        goto LABEL_23;
      }
      goto LABEL_20;
    }
    goto LABEL_26;
  }
  while ( 1 )
  {
    if ( (_BYTE)v5 != 5 )
      goto LABEL_30;
    v9 = *((_WORD *)v2 + 1);
    if ( v9 == 34 )
      break;
    if ( v9 != 49 )
    {
      if ( v9 != 50 )
        goto LABEL_30;
LABEL_8:
      if ( (v2[7] & 0x40) != 0 )
        v6 = (unsigned __int8 *)*((_QWORD *)v2 - 1);
      else
        v6 = &v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
      v2 = *(unsigned __int8 **)v6;
      goto LABEL_11;
    }
LABEL_20:
    if ( (v2[7] & 0x40) != 0 )
      v10 = (__int64 *)*((_QWORD *)v2 - 1);
    else
      v10 = (__int64 *)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
    v11 = *v10;
    if ( *(_BYTE *)(*(_QWORD *)(*v10 + 8) + 8LL) != 14 )
      goto LABEL_30;
LABEL_23:
    v2 = (unsigned __int8 *)v11;
    if ( !v23 )
    {
LABEL_24:
      a2 = (__int64)v2;
      sub_C8CC70(&v19, v2);
      v12 = v23;
      if ( !v13 )
        goto LABEL_31;
      goto LABEL_4;
    }
LABEL_12:
    v7 = v20;
    v8 = &v20[HIDWORD(v21)];
    if ( v20 != v8 )
    {
      while ( v2 != *v7 )
      {
        if ( v8 == ++v7 )
          goto LABEL_15;
      }
      return v2;
    }
LABEL_15:
    if ( HIDWORD(v21) >= (unsigned int)v21 )
      goto LABEL_24;
    ++HIDWORD(v21);
    *v8 = v2;
    ++v19;
    v5 = *v2;
    if ( (unsigned __int8)v5 > 0x1Cu )
      goto LABEL_5;
  }
LABEL_26:
  v14 = *((_DWORD *)v2 + 1) & 0x7FFFFFF;
  v15 = &v2[32 * (1 - v14)];
  if ( v2 == v15 )
  {
LABEL_35:
    v2 = *(unsigned __int8 **)&v2[-32 * v14];
LABEL_11:
    if ( !v23 )
      goto LABEL_24;
    goto LABEL_12;
  }
  while ( 1 )
  {
    v16 = *(_BYTE **)v15;
    if ( **(_BYTE **)v15 != 17 )
      break;
    v17 = *((_DWORD *)v16 + 8);
    if ( v17 <= 0x40 )
    {
      if ( *((_QWORD *)v16 + 3) )
        break;
    }
    else if ( v17 != (unsigned int)sub_C444A0(v16 + 24) )
    {
      break;
    }
    v15 += 32;
    if ( v2 == v15 )
      goto LABEL_35;
  }
LABEL_30:
  v12 = v23;
LABEL_31:
  if ( !v12 )
    _libc_free(v20, a2);
  return v2;
}
