// Function: sub_18DD490
// Address: 0x18dd490
//
__int64 __fastcall sub_18DD490(__int64 a1)
{
  unsigned int v1; // eax
  _QWORD *v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // r15
  _QWORD *v6; // rbx
  int v7; // r8d
  int v8; // r9d
  unsigned __int8 v9; // al
  _QWORD *v10; // rax
  char v11; // dl
  unsigned int v12; // r12d
  _QWORD *v13; // rsi
  _QWORD *v14; // rcx
  __int64 v15; // rax
  _QWORD *v17; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v18; // [rsp+8h] [rbp-E8h]
  _QWORD v19[8]; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v20; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v21; // [rsp+58h] [rbp-98h]
  _QWORD *v22; // [rsp+60h] [rbp-90h]
  __int64 v23; // [rsp+68h] [rbp-88h]
  int v24; // [rsp+70h] [rbp-80h]
  _QWORD v25[15]; // [rsp+78h] [rbp-78h] BYREF

  v21 = v25;
  v22 = v25;
  v17 = v19;
  v24 = 0;
  v20 = 1;
  v18 = 0x800000001LL;
  v19[0] = a1;
  v23 = 0x100000008LL;
  v1 = 1;
  v25[0] = a1;
  v2 = v19;
  while ( 1 )
  {
    v3 = v1--;
    v4 = v2[v3 - 1];
    LODWORD(v18) = v1;
    v5 = *(_QWORD *)(v4 + 8);
    if ( v5 )
      break;
LABEL_11:
    if ( !v1 )
    {
      v12 = 0;
      goto LABEL_26;
    }
  }
  while ( 1 )
  {
    v6 = sub_1648700(v5);
    v9 = *((_BYTE *)v6 + 16);
    if ( v9 > 0x17u )
      break;
LABEL_6:
    if ( *(_BYTE *)(v4 + 16) == 69 )
      goto LABEL_25;
    v10 = v21;
    if ( v22 == v21 )
    {
      v13 = &v21[HIDWORD(v23)];
      if ( v21 != v13 )
      {
        v14 = 0;
        while ( v6 != (_QWORD *)*v10 )
        {
          if ( *v10 == -2 )
            v14 = v10;
          if ( v13 == ++v10 )
          {
            if ( !v14 )
              goto LABEL_31;
            *v14 = v6;
            --v24;
            ++v20;
            goto LABEL_21;
          }
        }
        goto LABEL_9;
      }
LABEL_31:
      if ( HIDWORD(v23) < (unsigned int)v23 )
      {
        ++HIDWORD(v23);
        *v13 = v6;
        ++v20;
LABEL_21:
        v15 = (unsigned int)v18;
        if ( (unsigned int)v18 >= HIDWORD(v18) )
        {
          sub_16CD150((__int64)&v17, v19, 0, 8, v7, v8);
          v15 = (unsigned int)v18;
        }
        v17[v15] = v6;
        LODWORD(v18) = v18 + 1;
        goto LABEL_9;
      }
    }
    sub_16CCBA0((__int64)&v20, (__int64)v6);
    if ( v11 )
      goto LABEL_21;
LABEL_9:
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
    {
      v1 = v18;
      v2 = v17;
      goto LABEL_11;
    }
  }
  if ( v9 != 55 )
  {
    if ( v9 == 78 )
      goto LABEL_9;
    goto LABEL_6;
  }
  if ( (unsigned int)sub_1648720(v5) )
    goto LABEL_9;
LABEL_25:
  v2 = v17;
  v12 = 1;
LABEL_26:
  if ( v2 != v19 )
    _libc_free((unsigned __int64)v2);
  if ( v22 != v21 )
    _libc_free((unsigned __int64)v22);
  return v12;
}
