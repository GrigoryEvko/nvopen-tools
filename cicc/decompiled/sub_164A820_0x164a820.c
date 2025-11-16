// Function: sub_164A820
// Address: 0x164a820
//
__int64 __fastcall sub_164A820(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int8 v3; // al
  __int64 *v4; // r12
  __int64 *v5; // rax
  unsigned __int64 v6; // rdi
  __int64 *v7; // rax
  char v8; // dl
  __int16 v9; // ax
  __int64 *v10; // rsi
  __int64 *v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-68h] BYREF
  __int64 v16; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v17; // [rsp+18h] [rbp-58h]
  __int64 *v18; // [rsp+20h] [rbp-50h]
  __int64 v19; // [rsp+28h] [rbp-48h]
  int v20; // [rsp+30h] [rbp-40h]
  _QWORD v21[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return v1;
  v20 = 0;
  v17 = v21;
  v18 = v21;
  v19 = 0x100000004LL;
  v21[0] = a1;
  v16 = 1;
LABEL_4:
  v3 = *(_BYTE *)(v1 + 16);
  if ( v3 <= 0x17u )
  {
    while ( 1 )
    {
      if ( v3 != 5 )
      {
        if ( v3 == 1 )
          __asm { jmp     rax }
        goto LABEL_18;
      }
      v9 = *(_WORD *)(v1 + 18);
      if ( v9 != 32 )
        break;
LABEL_21:
      if ( (*(_BYTE *)(v1 + 17) & 2) == 0 )
        goto LABEL_18;
      if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
        goto LABEL_9;
LABEL_23:
      v4 = (__int64 *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF));
LABEL_10:
      v1 = *v4;
LABEL_11:
      v5 = v17;
      if ( v18 == v17 )
      {
        v10 = &v17[HIDWORD(v19)];
        if ( v17 != v10 )
        {
          v11 = 0;
          while ( v1 != *v5 )
          {
            if ( *v5 == -2 )
              v11 = v5;
            if ( v10 == ++v5 )
            {
              if ( !v11 )
                goto LABEL_38;
              *v11 = v1;
              --v20;
              ++v16;
              goto LABEL_4;
            }
          }
          return v1;
        }
LABEL_38:
        if ( HIDWORD(v19) < (unsigned int)v19 )
        {
          ++HIDWORD(v19);
          *v10 = v1;
          ++v16;
          goto LABEL_4;
        }
      }
      sub_16CCBA0(&v16, v1);
      v6 = (unsigned __int64)v18;
      v7 = v17;
      if ( !v8 )
        goto LABEL_19;
      v3 = *(_BYTE *)(v1 + 16);
      if ( v3 > 0x17u )
        goto LABEL_5;
    }
    if ( v9 == 47 || v9 == 48 )
    {
LABEL_8:
      if ( (*(_BYTE *)(v1 + 23) & 0x40) == 0 )
        goto LABEL_23;
LABEL_9:
      v4 = *(__int64 **)(v1 - 8);
      goto LABEL_10;
    }
  }
  else
  {
LABEL_5:
    switch ( v3 )
    {
      case 0x38u:
        goto LABEL_21;
      case 0x47u:
      case 0x48u:
        goto LABEL_8;
      case 0x4Eu:
        v12 = v1 | 4;
        v13 = v1 & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_35;
      case 0x1Du:
        v12 = v1 & 0xFFFFFFFFFFFFFFFBLL;
        v13 = v1 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_35:
        v15 = v12;
        if ( v13 )
        {
          v14 = sub_14AF150(&v15);
          if ( v14 )
          {
            v1 = v14;
            goto LABEL_11;
          }
        }
        break;
    }
  }
LABEL_18:
  v6 = (unsigned __int64)v18;
  v7 = v17;
LABEL_19:
  if ( (__int64 *)v6 == v7 )
    return v1;
  _libc_free(v6);
  return v1;
}
