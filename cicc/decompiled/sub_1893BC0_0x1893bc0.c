// Function: sub_1893BC0
// Address: 0x1893bc0
//
void __fastcall sub_1893BC0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 v5; // rax
  _BYTE *v6; // rdi
  __int64 v7; // rbx
  _QWORD *v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // r15
  char v11; // dl
  _QWORD *v12; // r14
  _QWORD *v13; // rax
  _QWORD *v14; // rsi
  _QWORD *v15; // rcx
  _BYTE *v16; // rsi
  _QWORD *v17; // [rsp+8h] [rbp-C8h] BYREF
  _BYTE *v18; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE *v19; // [rsp+18h] [rbp-B8h]
  _BYTE *v20; // [rsp+20h] [rbp-B0h]
  __int64 v21; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD *v22; // [rsp+38h] [rbp-98h]
  _QWORD *v23; // [rsp+40h] [rbp-90h]
  __int64 v24; // [rsp+48h] [rbp-88h]
  int v25; // [rsp+50h] [rbp-80h]
  _QWORD v26[15]; // [rsp+58h] [rbp-78h] BYREF

  v18 = 0;
  v19 = 0;
  v20 = 0;
  v3 = (_QWORD *)sub_22077B0(8);
  v4 = v3;
  if ( v3 )
  {
    *v3 = a2;
    v5 = a2;
  }
  else
  {
    v5 = MEMORY[0];
  }
  v6 = v4 + 1;
  v18 = v4;
  v20 = v4 + 1;
  v22 = v26;
  v23 = v26;
  v24 = 0x100000008LL;
  v25 = 0;
  v26[0] = a2;
  v21 = 1;
  while ( 1 )
  {
    v6 -= 8;
    v19 = v6;
    v7 = *(_QWORD *)(v5 + 8);
    if ( v7 )
    {
      while ( 1 )
      {
        v8 = sub_1648700(v7);
        v9 = *((_BYTE *)v8 + 16);
        if ( v9 <= 0x17u )
          break;
        sub_1893470(a1, *(_QWORD *)(v8[5] + 56LL));
LABEL_7:
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
        {
          v6 = v19;
          goto LABEL_28;
        }
      }
      if ( (unsigned __int8)(v9 - 4) > 0xCu )
        goto LABEL_7;
      v10 = v8[1];
      if ( !v10 )
        goto LABEL_7;
      while ( 1 )
      {
        v12 = sub_1648700(v10);
        v13 = v22;
        if ( v23 != v22 )
          goto LABEL_12;
        v14 = &v22[HIDWORD(v24)];
        if ( v22 != v14 )
        {
          v15 = 0;
          while ( v12 != (_QWORD *)*v13 )
          {
            if ( *v13 == -2 )
              v15 = v13;
            if ( v14 == ++v13 )
            {
              if ( !v15 )
                goto LABEL_30;
              *v15 = v12;
              --v25;
              ++v21;
              goto LABEL_13;
            }
          }
LABEL_21:
          v17 = v12;
          v16 = v19;
          if ( v19 == v20 )
          {
            sub_12879C0((__int64)&v18, v19, &v17);
          }
          else
          {
            if ( v19 )
            {
              *(_QWORD *)v19 = v12;
              v16 = v19;
            }
            v19 = v16 + 8;
          }
          goto LABEL_13;
        }
LABEL_30:
        if ( HIDWORD(v24) < (unsigned int)v24 )
        {
          ++HIDWORD(v24);
          *v14 = v12;
          ++v21;
        }
        else
        {
LABEL_12:
          sub_16CCBA0((__int64)&v21, (__int64)v12);
          if ( !v11 )
            goto LABEL_21;
        }
LABEL_13:
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          goto LABEL_7;
      }
    }
LABEL_28:
    if ( v18 == v6 )
      break;
    v5 = *((_QWORD *)v6 - 1);
  }
  if ( v23 != v22 )
  {
    _libc_free((unsigned __int64)v23);
    v6 = v18;
  }
  if ( v6 )
    j_j___libc_free_0(v6, v20 - v6);
}
