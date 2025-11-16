// Function: sub_BD4070
// Address: 0xbd4070
//
unsigned __int8 *__fastcall sub_BD4070(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  __int64 v4; // rbx
  int v5; // eax
  unsigned __int8 *v6; // r12
  char v7; // al
  char v8; // dl
  __int16 v9; // ax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int8 *v13; // rax
  unsigned __int8 **v14; // rax
  unsigned __int8 **v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int8 **v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  int v20; // [rsp+18h] [rbp-38h]
  char v21; // [rsp+1Ch] [rbp-34h]
  unsigned __int8 *v22; // [rsp+20h] [rbp-30h] BYREF

  v2 = a1;
  if ( *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) != 14 )
    return v2;
  v20 = 0;
  v4 = 0x8000000000041LL;
  v18 = &v22;
  v19 = 0x100000004LL;
  v21 = 1;
  v22 = a1;
  v17 = 1;
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
    v16 = (unsigned int)(v5 - 34);
    if ( (unsigned __int8)v16 > 0x33u )
      goto LABEL_24;
    if ( !_bittest64(&v4, v16) )
      goto LABEL_24;
    a2 = 52;
    v11 = sub_B494D0((__int64)v2, 52);
    if ( !v11 )
      goto LABEL_24;
LABEL_19:
    v2 = (unsigned __int8 *)v11;
LABEL_29:
    if ( v21 )
      goto LABEL_30;
    goto LABEL_11;
  }
  while ( 1 )
  {
    if ( (v2[7] & 0x40) != 0 )
      v6 = (unsigned __int8 *)*((_QWORD *)v2 - 1);
    else
      v6 = &v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
    v2 = *(unsigned __int8 **)v6;
    if ( v21 )
    {
LABEL_30:
      v14 = v18;
      v15 = &v18[HIDWORD(v19)];
      if ( v18 != v15 )
      {
        while ( v2 != *v14 )
        {
          if ( v15 == ++v14 )
            goto LABEL_33;
        }
        return v2;
      }
LABEL_33:
      if ( HIDWORD(v19) < (unsigned int)v19 )
      {
        ++HIDWORD(v19);
        *v15 = v2;
        ++v17;
        goto LABEL_4;
      }
    }
LABEL_11:
    a2 = (__int64)v2;
    sub_C8CC70(&v17, v2);
    v7 = v21;
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
LABEL_27:
    if ( (v2[1] & 2) == 0 )
      goto LABEL_24;
    v2 = *(unsigned __int8 **)&v2[-32 * v12];
    goto LABEL_29;
  }
  while ( **(_BYTE **)v13 == 17 )
  {
    v13 += 32;
    if ( v2 == v13 )
      goto LABEL_27;
  }
LABEL_24:
  v7 = v21;
LABEL_25:
  if ( v7 )
    return v2;
  _libc_free(v18, a2);
  return v2;
}
