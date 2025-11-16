// Function: sub_27DC8E0
// Address: 0x27dc8e0
//
void __fastcall sub_27DC8E0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // r13
  _BYTE *v8; // r15
  char v9; // di
  __int64 v10; // rsi
  __int64 *v11; // rax
  _BYTE *v12; // [rsp+0h] [rbp-240h] BYREF
  __int64 v13; // [rsp+8h] [rbp-238h]
  _BYTE v14[560]; // [rsp+10h] [rbp-230h] BYREF

  v12 = v14;
  v13 = 0x2000000000LL;
  sub_D0E1D0(a2, (__int64)&v12);
  v7 = (unsigned __int64)v12;
  v8 = &v12[16 * (unsigned int)v13];
  if ( v8 == v12 )
    goto LABEL_10;
  v9 = *(_BYTE *)(a1 + 124);
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v7 + 8);
        if ( v9 )
          break;
LABEL_13:
        v7 += 16LL;
        sub_C8CC70(a1 + 96, v10, (__int64)v3, v4, v5, v6);
        v9 = *(_BYTE *)(a1 + 124);
        if ( (_BYTE *)v7 == v8 )
          goto LABEL_9;
      }
      v11 = *(__int64 **)(a1 + 104);
      v4 = *(unsigned int *)(a1 + 116);
      v3 = &v11[v4];
      if ( v11 != v3 )
        break;
LABEL_15:
      if ( (unsigned int)v4 >= *(_DWORD *)(a1 + 112) )
        goto LABEL_13;
      v4 = (unsigned int)(v4 + 1);
      v7 += 16LL;
      *(_DWORD *)(a1 + 116) = v4;
      *v3 = v10;
      v9 = *(_BYTE *)(a1 + 124);
      ++*(_QWORD *)(a1 + 96);
      if ( (_BYTE *)v7 == v8 )
        goto LABEL_9;
    }
    while ( v10 != *v11 )
    {
      if ( v3 == ++v11 )
        goto LABEL_15;
    }
    v7 += 16LL;
  }
  while ( (_BYTE *)v7 != v8 );
LABEL_9:
  v7 = (unsigned __int64)v12;
LABEL_10:
  if ( (_BYTE *)v7 != v14 )
    _libc_free(v7);
}
