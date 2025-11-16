// Function: sub_3183650
// Address: 0x3183650
//
__int64 __fastcall sub_3183650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // eax
  _QWORD *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r14
  char *v11; // rbx
  char v12; // al
  __int64 *v13; // rax
  unsigned int v14; // r13d
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  _QWORD *v18; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v19; // [rsp+8h] [rbp-D8h]
  _QWORD v20[8]; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v21; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v22; // [rsp+58h] [rbp-88h]
  __int64 v23; // [rsp+60h] [rbp-80h]
  int v24; // [rsp+68h] [rbp-78h]
  char v25; // [rsp+6Ch] [rbp-74h]
  __int64 v26; // [rsp+70h] [rbp-70h] BYREF

  v22 = &v26;
  v18 = v20;
  v24 = 0;
  v25 = 1;
  v21 = 1;
  v19 = 0x800000001LL;
  v20[0] = a1;
  v23 = 0x100000008LL;
  v6 = 1;
  v26 = a1;
  v7 = v20;
  while ( 1 )
  {
    v8 = v6--;
    v9 = v7[v8 - 1];
    LODWORD(v19) = v6;
    v10 = *(_QWORD *)(v9 + 16);
    if ( v10 )
      break;
LABEL_14:
    if ( !v6 )
    {
      v14 = 0;
      goto LABEL_24;
    }
  }
  while ( 1 )
  {
    v11 = *(char **)(v10 + 24);
    v12 = *v11;
    if ( (unsigned __int8)*v11 > 0x1Cu )
      break;
LABEL_6:
    if ( *(_BYTE *)v9 == 76 )
      goto LABEL_23;
    if ( v25 )
    {
      v13 = v22;
      a4 = HIDWORD(v23);
      v8 = (__int64)&v22[HIDWORD(v23)];
      if ( v22 != (__int64 *)v8 )
      {
        while ( v11 != (char *)*v13 )
        {
          if ( (__int64 *)v8 == ++v13 )
            goto LABEL_16;
        }
        goto LABEL_12;
      }
LABEL_16:
      if ( HIDWORD(v23) < (unsigned int)v23 )
      {
        ++HIDWORD(v23);
        *(_QWORD *)v8 = v11;
        ++v21;
        goto LABEL_18;
      }
    }
    sub_C8CC70((__int64)&v21, *(_QWORD *)(v10 + 24), v8, a4, a5, a6);
    if ( (_BYTE)v8 )
    {
LABEL_18:
      v15 = (unsigned int)v19;
      a4 = HIDWORD(v19);
      v16 = (unsigned int)v19 + 1LL;
      if ( v16 > HIDWORD(v19) )
      {
        sub_C8D5F0((__int64)&v18, v20, v16, 8u, a5, a6);
        v15 = (unsigned int)v19;
      }
      v8 = (__int64)v18;
      v18[v15] = v11;
      LODWORD(v19) = v19 + 1;
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
      {
LABEL_13:
        v6 = v19;
        v7 = v18;
        goto LABEL_14;
      }
    }
    else
    {
LABEL_12:
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_13;
    }
  }
  if ( v12 != 62 )
  {
    if ( v12 == 85 )
      goto LABEL_12;
    goto LABEL_6;
  }
  if ( (unsigned int)sub_BD2910(v10) )
    goto LABEL_12;
LABEL_23:
  v7 = v18;
  v14 = 1;
LABEL_24:
  if ( v7 != v20 )
    _libc_free((unsigned __int64)v7);
  if ( !v25 )
    _libc_free((unsigned __int64)v22);
  return v14;
}
