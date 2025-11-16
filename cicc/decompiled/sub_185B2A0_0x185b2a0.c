// Function: sub_185B2A0
// Address: 0x185b2a0
//
_BOOL8 __fastcall sub_185B2A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, int a6)
{
  int v6; // r12d
  __int64 v7; // rax
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int8 v13; // dl
  _BOOL4 v14; // r15d
  __int64 *v15; // rbx
  __int64 v16; // r15
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 *v21; // [rsp+8h] [rbp-68h]
  _QWORD *v22; // [rsp+10h] [rbp-60h] BYREF
  __int64 v23; // [rsp+18h] [rbp-58h]
  _QWORD v24[10]; // [rsp+20h] [rbp-50h] BYREF

  v6 = 20;
  v7 = *(_QWORD *)(a1 + 24);
  v22 = v24;
  v8 = v24;
  v24[0] = v7;
  v23 = 0x400000001LL;
  v9 = 1;
  while ( 1 )
  {
    v10 = v9;
    v11 = v9 - 1;
    v12 = v8[v10 - 1];
    LODWORD(v23) = v11;
    v13 = *(_BYTE *)(v12 + 8);
    if ( v13 == 15 )
      break;
    if ( v13 > 0xFu )
    {
      if ( v13 != 16 )
        goto LABEL_19;
      goto LABEL_16;
    }
    if ( v13 != 13 )
    {
      if ( v13 != 14 )
        goto LABEL_19;
LABEL_16:
      v19 = *(_QWORD *)(v12 + 24);
      if ( (unsigned int)v11 >= HIDWORD(v23) )
      {
        sub_16CD150((__int64)&v22, v24, 0, 8, (int)a5, a6);
        v8 = v22;
        v11 = (unsigned int)v23;
      }
      v8[v11] = v19;
      v8 = v22;
      LODWORD(v23) = v23 + 1;
      goto LABEL_19;
    }
    v14 = (*(_DWORD *)(v12 + 8) & 0x100) == 0;
    if ( (*(_DWORD *)(v12 + 8) & 0x100) == 0 )
      goto LABEL_22;
    v15 = *(__int64 **)(v12 + 16);
    a5 = &v15[*(unsigned int *)(v12 + 12)];
    if ( v15 != a5 )
    {
      do
      {
        v16 = *v15;
        v17 = *(unsigned __int8 *)(*v15 + 8);
        if ( (_BYTE)v17 == 15 )
        {
          v8 = v22;
          v14 = 1;
          goto LABEL_22;
        }
        if ( (unsigned int)(v17 - 13) <= 1 || v17 == 16 )
        {
          v18 = (unsigned int)v23;
          if ( (unsigned int)v23 >= HIDWORD(v23) )
          {
            v21 = a5;
            sub_16CD150((__int64)&v22, v24, 0, 8, (int)a5, a6);
            v18 = (unsigned int)v23;
            a5 = v21;
          }
          v22[v18] = v16;
          LODWORD(v23) = v23 + 1;
        }
        ++v15;
      }
      while ( a5 != v15 );
      v8 = v22;
    }
LABEL_19:
    if ( !--v6 )
      break;
    v9 = v23;
    if ( !(_DWORD)v23 )
    {
      v14 = 0;
      goto LABEL_22;
    }
  }
  v14 = 1;
LABEL_22:
  if ( v8 != v24 )
    _libc_free((unsigned __int64)v8);
  return v14;
}
