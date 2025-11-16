// Function: sub_28ED4B0
// Address: 0x28ed4b0
//
void __fastcall sub_28ED4B0(__int64 a1, char *a2)
{
  unsigned __int8 v4; // al
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  unsigned __int8 **v7; // rax
  unsigned __int8 *v8; // r13
  char *v9; // r14
  unsigned __int8 v10; // al
  __int64 v11; // rsi
  bool v12; // zf
  __int64 v13; // rdx
  _BYTE *v14; // rax
  unsigned __int8 *v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // rax
  unsigned int v18; // eax
  __int64 v19; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v20; // [rsp+8h] [rbp-28h]

  *(_DWORD *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  v4 = *a2;
  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 32) = 0;
  if ( v4 > 0x1Cu && (unsigned __int8)(v4 - 57) <= 1u )
  {
    if ( (a2[7] & 0x40) != 0 )
      v7 = (unsigned __int8 **)*((_QWORD *)a2 - 1);
    else
      v7 = (unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v8 = *v7;
    v9 = (char *)v7[4];
    v10 = **v7;
    v11 = (__int64)(v8 + 24);
    if ( v10 != 17 )
    {
      v13 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v8 + 1) + 8LL) - 17;
      if ( (unsigned int)v13 <= 1 && v10 <= 0x15u )
      {
        v14 = sub_AD7630((__int64)v8, 0, v13);
        if ( v14 )
        {
          if ( *v14 == 17 )
          {
            v15 = v8;
            v8 = (unsigned __int8 *)v9;
            v9 = (char *)v15;
          }
        }
      }
      v16 = (unsigned __int8)*v9;
      if ( (_BYTE)v16 == 17 )
      {
        v18 = *(_DWORD *)(a1 + 24);
        v11 = (__int64)(v9 + 24);
      }
      else
      {
        if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v9 + 1) + 8LL) - 17 > 1 )
          goto LABEL_3;
        if ( (unsigned __int8)v16 > 0x15u )
          goto LABEL_3;
        v17 = sub_AD7630((__int64)v9, 0, v16);
        if ( !v17 || *v17 != 17 )
          goto LABEL_3;
        v11 = (__int64)(v17 + 24);
        v18 = *(_DWORD *)(a1 + 24);
      }
      v9 = (char *)v8;
      if ( v18 > 0x40 )
        goto LABEL_13;
    }
    if ( *(_DWORD *)(v11 + 8) <= 0x40u )
    {
      *(_QWORD *)(a1 + 16) = *(_QWORD *)v11;
      *(_DWORD *)(a1 + 24) = *(_DWORD *)(v11 + 8);
    }
    else
    {
LABEL_13:
      sub_C43990(a1 + 16, v11);
    }
    v12 = *a2 == 58;
    *(_QWORD *)(a1 + 8) = v9;
    *(_BYTE *)(a1 + 36) = v12;
    return;
  }
LABEL_3:
  *(_QWORD *)(a1 + 8) = a2;
  v20 = sub_BCB060(*((_QWORD *)a2 + 1));
  if ( v20 > 0x40 )
    sub_C43690((__int64)&v19, 0, 0);
  else
    v19 = 0;
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
  {
    v5 = *(_QWORD *)(a1 + 16);
    if ( v5 )
      j_j___libc_free_0_0(v5);
  }
  v6 = v19;
  *(_BYTE *)(a1 + 36) = 1;
  *(_QWORD *)(a1 + 16) = v6;
  *(_DWORD *)(a1 + 24) = v20;
}
