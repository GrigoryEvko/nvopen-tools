// Function: sub_2DB32C0
// Address: 0x2db32c0
//
void __fastcall sub_2DB32C0(_QWORD *a1, _BYTE *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE **v6; // r8
  _BYTE *v8; // r12
  char v9; // bl
  unsigned int v10; // r15d
  __int64 (*v11)(); // rax
  __int64 v12; // rbx
  __int64 i; // r12
  _BYTE *v14; // rdi
  _BYTE *v15; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v16; // [rsp+18h] [rbp-D8h]
  _BYTE dest[208]; // [rsp+20h] [rbp-D0h] BYREF

  v6 = &v15;
  v8 = a2;
  v9 = a3;
  v10 = *((_DWORD *)a1 + 84);
  v15 = dest;
  v16 = 0x400000000LL;
  if ( v10 )
  {
    if ( v10 > 4 )
    {
      a2 = dest;
      sub_C8D5F0((__int64)&v15, dest, v10, 0x28u, (__int64)&v15, a6);
      v14 = v15;
      v6 = &v15;
      a3 = 40LL * *((unsigned int *)a1 + 84);
      if ( !a3 )
        goto LABEL_21;
    }
    else
    {
      v14 = dest;
      a3 = 40LL * v10;
    }
    a2 = (_BYTE *)a1[41];
    memcpy(v14, a2, a3);
    v6 = &v15;
LABEL_21:
    LODWORD(v16) = v10;
  }
  if ( v9 )
  {
    v11 = *(__int64 (**)())(*(_QWORD *)*a1 + 880LL);
    if ( v11 != sub_2DB1B20 )
    {
      a2 = &v15;
      ((void (__fastcall *)(_QWORD, _BYTE **))v11)(*a1, &v15);
    }
  }
  v12 = *((_QWORD *)v8 + 7);
  for ( i = sub_2E313E0(v8, a2, a3, a4, v6); v12 != i; v12 = *(_QWORD *)(v12 + 8) )
  {
    while ( 1 )
    {
      if ( (unsigned __int16)(*(_WORD *)(v12 + 68) - 14) > 4u )
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD))(*(_QWORD *)*a1 + 968LL))(
          *a1,
          v12,
          v15,
          (unsigned int)v16);
      if ( (*(_BYTE *)v12 & 4) == 0 )
        break;
      v12 = *(_QWORD *)(v12 + 8);
      if ( v12 == i )
        goto LABEL_12;
    }
    while ( (*(_BYTE *)(v12 + 44) & 8) != 0 )
      v12 = *(_QWORD *)(v12 + 8);
  }
LABEL_12:
  if ( v15 != dest )
    _libc_free((unsigned __int64)v15);
}
