// Function: sub_BD3990
// Address: 0xbd3990
//
unsigned __int8 *__fastcall sub_BD3990(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  __int64 v4; // r13
  int v5; // eax
  unsigned __int8 *v6; // r12
  char v7; // al
  char v8; // dl
  __int16 v9; // ax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r14
  unsigned __int8 *v13; // rbx
  _BYTE *v14; // rdi
  unsigned int v15; // r15d
  unsigned __int8 **v16; // rax
  unsigned __int8 **v17; // rdx
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
  if ( (unsigned __int8)v5 <= 0x1Cu )
    goto LABEL_13;
LABEL_5:
  if ( (_BYTE)v5 == 63 )
    goto LABEL_20;
  if ( (unsigned __int8)v5 == 78 )
  {
LABEL_16:
    if ( (v2[7] & 0x40) != 0 )
      v10 = (__int64 *)*((_QWORD *)v2 - 1);
    else
      v10 = (__int64 *)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
    v11 = *v10;
    if ( *(_BYTE *)(*(_QWORD *)(*v10 + 8) + 8LL) != 14 )
      goto LABEL_24;
    goto LABEL_19;
  }
  if ( (unsigned __int8)v5 != 79 )
  {
    v18 = (unsigned int)(v5 - 34);
    if ( (unsigned __int8)v18 > 0x33u )
      goto LABEL_24;
    if ( !_bittest64(&v4, v18) )
      goto LABEL_24;
    a2 = 52;
    v11 = sub_B494D0((__int64)v2, 52);
    if ( !v11 )
      goto LABEL_24;
LABEL_19:
    v2 = (unsigned __int8 *)v11;
LABEL_30:
    if ( v23 )
      goto LABEL_31;
    goto LABEL_11;
  }
  while ( 1 )
  {
    if ( (v2[7] & 0x40) != 0 )
      v6 = (unsigned __int8 *)*((_QWORD *)v2 - 1);
    else
      v6 = &v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
    v2 = *(unsigned __int8 **)v6;
    if ( v23 )
    {
LABEL_31:
      v16 = v20;
      v17 = &v20[HIDWORD(v21)];
      if ( v20 != v17 )
      {
        while ( v2 != *v16 )
        {
          if ( v17 == ++v16 )
            goto LABEL_34;
        }
        return v2;
      }
LABEL_34:
      if ( HIDWORD(v21) < (unsigned int)v21 )
      {
        ++HIDWORD(v21);
        *v17 = v2;
        ++v19;
        goto LABEL_4;
      }
    }
LABEL_11:
    a2 = (__int64)v2;
    sub_C8CC70(&v19, v2);
    v7 = v23;
    if ( !v8 )
      goto LABEL_25;
    v5 = *v2;
    if ( (unsigned __int8)v5 > 0x1Cu )
      goto LABEL_5;
LABEL_13:
    if ( (_BYTE)v5 != 5 )
      goto LABEL_24;
    v9 = *((_WORD *)v2 + 1);
    if ( v9 == 34 )
      break;
    if ( v9 == 49 )
      goto LABEL_16;
    if ( v9 != 50 )
      goto LABEL_24;
  }
LABEL_20:
  v12 = *((_DWORD *)v2 + 1) & 0x7FFFFFF;
  v13 = &v2[32 * (1 - v12)];
  if ( v2 == v13 )
  {
LABEL_29:
    v2 = *(unsigned __int8 **)&v2[-32 * v12];
    goto LABEL_30;
  }
  while ( 1 )
  {
    v14 = *(_BYTE **)v13;
    if ( **(_BYTE **)v13 != 17 )
      break;
    v15 = *((_DWORD *)v14 + 8);
    if ( v15 <= 0x40 )
    {
      if ( *((_QWORD *)v14 + 3) )
        break;
    }
    else if ( v15 != (unsigned int)sub_C444A0(v14 + 24) )
    {
      break;
    }
    v13 += 32;
    if ( v2 == v13 )
      goto LABEL_29;
  }
LABEL_24:
  v7 = v23;
LABEL_25:
  if ( !v7 )
    _libc_free(v20, a2);
  return v2;
}
