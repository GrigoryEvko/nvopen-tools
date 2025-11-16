// Function: sub_2509520
// Address: 0x2509520
//
__int64 __fastcall sub_2509520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // rdi
  __int64 v8; // r8
  unsigned int v9; // eax
  unsigned int v10; // r13d
  _BYTE *v11; // rdi
  unsigned __int8 *v12; // r8
  unsigned __int8 *v13; // rdi
  __int64 v14; // r13
  unsigned __int8 *v15; // rdi
  unsigned __int8 *v16; // rax
  int v18; // eax
  char *v19; // [rsp+0h] [rbp-40h] BYREF
  unsigned int *v20; // [rsp+8h] [rbp-38h] BYREF
  __int64 v21; // [rsp+10h] [rbp-30h]
  _BYTE v22[40]; // [rsp+18h] [rbp-28h] BYREF

  v7 = *(char **)a2;
  v8 = *(unsigned int *)(a2 + 16);
  v20 = (unsigned int *)v22;
  v21 = 0;
  v19 = v7;
  if ( (_DWORD)v8 )
  {
    a2 += 8;
    sub_2506900((__int64)&v20, (char **)a2, a3, a4, v8, a6);
    if ( (_DWORD)v21 )
      goto LABEL_3;
    v7 = v19;
  }
  LOBYTE(v9) = sub_B491E0((__int64)v7);
  v10 = v9;
  if ( !(_BYTE)v9 )
  {
    v12 = (unsigned __int8 *)*((_QWORD *)v19 - 4);
    if ( !v12 )
      goto LABEL_23;
LABEL_4:
    if ( *sub_BD3990(v12, a2) )
      goto LABEL_9;
    v13 = (unsigned __int8 *)v19;
    v14 = *((_QWORD *)v19 + 1);
    if ( !(_DWORD)v21 )
    {
      if ( !sub_B491E0((__int64)v19) )
      {
        v15 = (unsigned __int8 *)*((_QWORD *)v19 - 4);
        if ( !v15 )
          goto LABEL_26;
        goto LABEL_7;
      }
      v13 = (unsigned __int8 *)v19;
    }
    v15 = *(unsigned __int8 **)&v13[32 * (*v20 - (unsigned __int64)(*((_DWORD *)v13 + 1) & 0x7FFFFFF))];
    if ( !v15 )
      goto LABEL_26;
LABEL_7:
    v16 = sub_BD3990(v15, a2);
    if ( !*v16 )
    {
      if ( v14 == **(_QWORD **)(*((_QWORD *)v16 + 3) + 16LL)
        && *(_QWORD *)(*((_QWORD *)v19 - 4) + 8LL) == *(_QWORD *)(*(_QWORD *)a1 + 8LL)
        && (unsigned int)sub_25093E0((unsigned __int8 **)&v19) == *(_QWORD *)(*(_QWORD *)a1 + 104LL)
        && !(_DWORD)v21 )
      {
        LOBYTE(v18) = sub_B49200((__int64)v19);
        v10 = v18 ^ 1;
LABEL_23:
        v11 = v20;
        goto LABEL_10;
      }
LABEL_9:
      v11 = v20;
      v10 = 0;
      goto LABEL_10;
    }
LABEL_26:
    BUG();
  }
LABEL_3:
  v11 = v20;
  v12 = *(unsigned __int8 **)&v19[32 * (*v20 - (unsigned __int64)(*((_DWORD *)v19 + 1) & 0x7FFFFFF))];
  if ( v12 )
    goto LABEL_4;
  v10 = 0;
LABEL_10:
  if ( v11 != v22 )
    _libc_free((unsigned __int64)v11);
  return v10;
}
