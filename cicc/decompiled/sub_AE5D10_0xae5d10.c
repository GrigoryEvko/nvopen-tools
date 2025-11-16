// Function: sub_AE5D10
// Address: 0xae5d10
//
__int64 __fastcall sub_AE5D10(__int64 a1, __int64 (__fastcall *a2)(__int64), __int64 a3)
{
  unsigned int v3; // r15d
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int8 v8; // al
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rdi
  __int64 v12; // r12
  __int64 v14; // [rsp+0h] [rbp-70h]
  _QWORD *v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h]
  _QWORD v17[10]; // [rsp+20h] [rbp-50h] BYREF

  v3 = 1;
  v15 = v17;
  v16 = 0x400000001LL;
  v17[0] = 0;
  while ( 1 )
  {
    v8 = *(_BYTE *)(a1 - 16);
    if ( (v8 & 2) == 0 )
      break;
    if ( v3 >= *(_DWORD *)(a1 - 24) )
      goto LABEL_15;
    if ( !*(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL * v3) )
    {
LABEL_12:
      v9 = (unsigned int)v16;
      v10 = (unsigned int)v16 + 1LL;
      if ( v10 > HIDWORD(v16) )
      {
        sub_C8D5F0(&v15, v17, v10, 8);
        v9 = (unsigned int)v16;
      }
      v15[v9] = 0;
      LODWORD(v16) = v16 + 1;
      goto LABEL_8;
    }
LABEL_4:
    v6 = a2(a3);
    if ( v6 )
    {
      v7 = (unsigned int)v16;
      if ( (unsigned __int64)(unsigned int)v16 + 1 > HIDWORD(v16) )
      {
        v14 = v6;
        sub_C8D5F0(&v15, v17, (unsigned int)v16 + 1LL, 8);
        v7 = (unsigned int)v16;
        v6 = v14;
      }
      v15[v7] = v6;
      LODWORD(v16) = v16 + 1;
    }
LABEL_8:
    ++v3;
  }
  if ( v3 < ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) )
  {
    if ( !*(_QWORD *)(a1 + -16 - 8LL * ((v8 >> 2) & 0xF) + 8LL * v3) )
      goto LABEL_12;
    goto LABEL_4;
  }
LABEL_15:
  v11 = (_QWORD *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v11 = (_QWORD *)*v11;
  v12 = sub_B9C770(v11, v15, (unsigned int)v16, 1, 1);
  sub_BA6610(v12, 0, v12);
  if ( v15 != v17 )
    _libc_free(v15, 0);
  return v12;
}
