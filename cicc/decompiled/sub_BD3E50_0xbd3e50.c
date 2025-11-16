// Function: sub_BD3E50
// Address: 0xbd3e50
//
unsigned __int8 *__fastcall sub_BD3E50(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  __int64 v4; // r13
  int v5; // eax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int8 *i; // rbx
  _BYTE *v10; // rdi
  unsigned int v11; // r15d
  char v12; // al
  unsigned __int8 **v13; // rax
  unsigned __int8 **v14; // rdx
  __int16 v15; // ax
  char v16; // dl
  unsigned __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int8 **v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+10h] [rbp-60h]
  int v21; // [rsp+18h] [rbp-58h]
  char v22; // [rsp+1Ch] [rbp-54h]
  unsigned __int8 *v23; // [rsp+20h] [rbp-50h] BYREF

  v2 = a1;
  if ( *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) == 14 )
  {
    v21 = 0;
    v4 = 0x8000000000041LL;
    v19 = &v23;
    v20 = 0x100000004LL;
    v22 = 1;
    v23 = a1;
    v18 = 1;
LABEL_4:
    v5 = *v2;
    if ( (unsigned __int8)v5 > 0x1Cu )
    {
LABEL_5:
      if ( (_BYTE)v5 == 63 )
        goto LABEL_11;
      if ( (_BYTE)v5 == 78 )
      {
LABEL_7:
        if ( (v2[7] & 0x40) != 0 )
          v6 = (__int64 *)*((_QWORD *)v2 - 1);
        else
          v6 = (__int64 *)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
        v7 = *v6;
        if ( *(_BYTE *)(*(_QWORD *)(*v6 + 8) + 8LL) != 14 )
          goto LABEL_15;
      }
      else
      {
        v17 = (unsigned int)(v5 - 34);
        if ( (unsigned __int8)v17 > 0x33u )
          goto LABEL_15;
        if ( !_bittest64(&v4, v17) )
          goto LABEL_15;
        a2 = 52;
        v7 = sub_B494D0((__int64)v2, 52);
        if ( !v7 )
          goto LABEL_15;
      }
      v2 = (unsigned __int8 *)v7;
      goto LABEL_21;
    }
    while ( 1 )
    {
      if ( (_BYTE)v5 != 5 )
        goto LABEL_15;
      v15 = *((_WORD *)v2 + 1);
      if ( v15 != 34 )
        break;
LABEL_11:
      v8 = *((_DWORD *)v2 + 1) & 0x7FFFFFF;
      for ( i = &v2[32 * (1 - v8)]; v2 != i; i += 32 )
      {
        v10 = *(_BYTE **)i;
        if ( **(_BYTE **)i != 17 )
          goto LABEL_15;
        v11 = *((_DWORD *)v10 + 8);
        if ( v11 <= 0x40 )
        {
          if ( *((_QWORD *)v10 + 3) )
            goto LABEL_15;
        }
        else if ( v11 != (unsigned int)sub_C444A0(v10 + 24) )
        {
          goto LABEL_15;
        }
      }
      v2 = *(unsigned __int8 **)&v2[-32 * v8];
LABEL_21:
      if ( !v22 )
      {
LABEL_31:
        a2 = (__int64)v2;
        sub_C8CC70(&v18, v2);
        v12 = v22;
        if ( !v16 )
          goto LABEL_16;
        goto LABEL_4;
      }
      v13 = v19;
      v14 = &v19[HIDWORD(v20)];
      if ( v19 != v14 )
      {
        while ( v2 != *v13 )
        {
          if ( v14 == ++v13 )
            goto LABEL_25;
        }
        return v2;
      }
LABEL_25:
      if ( HIDWORD(v20) >= (unsigned int)v20 )
        goto LABEL_31;
      ++HIDWORD(v20);
      *v14 = v2;
      ++v18;
      v5 = *v2;
      if ( (unsigned __int8)v5 > 0x1Cu )
        goto LABEL_5;
    }
    if ( v15 == 49 )
      goto LABEL_7;
LABEL_15:
    v12 = v22;
LABEL_16:
    if ( !v12 )
      _libc_free(v19, a2);
  }
  return v2;
}
