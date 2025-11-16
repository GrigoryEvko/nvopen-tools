// Function: sub_1C4B2D0
// Address: 0x1c4b2d0
//
__int64 __fastcall sub_1C4B2D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // rdx
  __int64 v10; // r14
  int v11; // r15d
  unsigned int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // eax
  _QWORD *v18; // rdi
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned int i; // eax
  unsigned int v23; // eax
  __int64 v24; // [rsp+8h] [rbp-68h]
  _QWORD *v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  _QWORD v27[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = 0;
  v26 = 0x200000000LL;
  v7 = *(_QWORD **)a1;
  v25 = v27;
  v8 = *v7;
  if ( *(_BYTE *)(*v7 + 16LL) <= 0x17u )
    return v6;
  v9 = (*(_BYTE *)(v8 + 23) & 0x40) != 0
     ? *(__int64 **)(v8 - 8)
     : (__int64 *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
  v10 = *v9;
  v11 = *(_DWORD *)(a1 + 8);
  v6 = 1;
  LODWORD(v26) = 1;
  v27[0] = v10;
  if ( v11 == 1 )
    return v6;
  v12 = 1;
  v13 = v7[1];
  if ( *(_BYTE *)(v13 + 16) <= 0x17u )
  {
LABEL_12:
    v18 = v25;
    v6 = 0;
    goto LABEL_13;
  }
  while ( 1 )
  {
    if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
    {
      v14 = **(_QWORD **)(v13 - 8);
      v15 = (unsigned int)v26;
      if ( (unsigned int)v26 < HIDWORD(v26) )
        goto LABEL_8;
    }
    else
    {
      v14 = *(_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
      v15 = (unsigned int)v26;
      if ( (unsigned int)v26 < HIDWORD(v26) )
        goto LABEL_8;
    }
    v24 = v14;
    sub_16CD150((__int64)&v25, v27, 0, 8, v14, a6);
    v15 = (unsigned int)v26;
    v14 = v24;
LABEL_8:
    if ( v10 != v14 )
      v6 = 0;
    ++v12;
    v25[v15] = v14;
    v16 = (unsigned int)v26;
    v17 = v26 + 1;
    LODWORD(v26) = v26 + 1;
    if ( v11 == v12 )
      break;
    v13 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * v12);
    if ( *(_BYTE *)(v13 + 16) <= 0x17u )
      goto LABEL_12;
  }
  v18 = v25;
  if ( !(_BYTE)v6 )
  {
    v20 = *(unsigned __int8 *)(*v25 + 16LL);
    if ( (unsigned __int8)v20 > 0x17u )
    {
      v21 = (unsigned int)(v20 - 53);
      LOBYTE(a6) = (unsigned __int8)(v20 - 53) <= 1u;
      LOBYTE(v21) = a6 | ((_BYTE)v20 == 56);
      if ( (_BYTE)v21 )
      {
        if ( v17 != 1 )
        {
          for ( i = 1; ; ++i )
          {
            if ( (_BYTE)v20 == 53 )
            {
              if ( *(_BYTE *)(v25[i] + 16LL) != 53 )
                goto LABEL_13;
            }
            else if ( (_BYTE)v20 == 54 )
            {
              if ( *(_BYTE *)(v25[i] + 16LL) != 54 )
                goto LABEL_13;
            }
            else if ( (_BYTE)v20 != 56 || *(_BYTE *)(v25[i] + 16LL) != 56 )
            {
              goto LABEL_13;
            }
            a6 = i + 1;
            if ( (_DWORD)v16 == i )
              break;
          }
        }
        v6 = v21;
        if ( (_BYTE)v20 != 53 )
        {
          v23 = sub_1C4B2D0(&v25, v27, v16, v20, v21, a6);
          v18 = v25;
          v6 = v23;
        }
      }
    }
  }
LABEL_13:
  if ( v18 != v27 )
    _libc_free((unsigned __int64)v18);
  return v6;
}
