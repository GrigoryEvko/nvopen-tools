// Function: sub_3544200
// Address: 0x3544200
//
unsigned __int64 __fastcall sub_3544200(__int64 a1, int a2)
{
  __int64 v3; // rdi
  unsigned __int64 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // r13
  char v9; // di
  unsigned __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-80h] BYREF
  unsigned __int64 *v14; // [rsp+8h] [rbp-78h]
  __int64 v15; // [rsp+10h] [rbp-70h]
  int v16; // [rsp+18h] [rbp-68h]
  char v17; // [rsp+1Ch] [rbp-64h]
  char v18; // [rsp+20h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v14 = (unsigned __int64 *)&v18;
  v13 = 0;
  v15 = 8;
  v16 = 0;
  v17 = 1;
  v8 = sub_2EBEE10(v3, a2);
LABEL_2:
  v9 = v17;
  if ( *(_WORD *)(v8 + 68) != 68 && *(_WORD *)(v8 + 68) )
  {
LABEL_4:
    if ( !v9 )
      _libc_free((unsigned __int64)v14);
    return v8;
  }
  while ( 1 )
  {
    if ( !v9 )
      goto LABEL_14;
    v11 = v14;
    v5 = HIDWORD(v15);
    v4 = &v14[HIDWORD(v15)];
    if ( v14 != v4 )
      break;
LABEL_13:
    if ( HIDWORD(v15) < (unsigned int)v15 )
    {
      ++HIDWORD(v15);
      *v4 = v8;
      v9 = v17;
      ++v13;
      goto LABEL_15;
    }
LABEL_14:
    sub_C8CC70((__int64)&v13, v8, (__int64)v4, v5, v6, v7);
    v9 = v17;
    if ( !(_BYTE)v4 )
      goto LABEL_4;
LABEL_15:
    v5 = *(_DWORD *)(v8 + 40) & 0xFFFFFF;
    if ( (unsigned int)v5 > 1 )
    {
      v6 = *(_QWORD *)(v8 + 32);
      v12 = 1;
      v4 = (unsigned __int64 *)(v6 + 104);
      while ( *(_QWORD *)(a1 + 904) != *v4 )
      {
        v12 = (unsigned int)(v12 + 2);
        v4 += 10;
        if ( (unsigned int)v12 >= (unsigned int)v5 )
          goto LABEL_7;
      }
      v8 = sub_2EBEE10(*(_QWORD *)(a1 + 40), *(_DWORD *)(v6 + 40 * v12 + 8));
      goto LABEL_2;
    }
LABEL_7:
    if ( *(_WORD *)(v8 + 68) && *(_WORD *)(v8 + 68) != 68 )
      goto LABEL_4;
  }
  while ( v8 != *v11 )
  {
    if ( v4 == ++v11 )
      goto LABEL_13;
  }
  return v8;
}
