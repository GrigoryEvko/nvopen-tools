// Function: sub_33CEF80
// Address: 0x33cef80
//
void __fastcall sub_33CEF80(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // r14
  _QWORD *v5; // rdi
  char v6; // al
  __int64 v7; // r9
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rdx
  int v14; // [rsp+8h] [rbp-B8h]
  _QWORD *v15; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-A8h]
  unsigned int v17; // [rsp+1Ch] [rbp-A4h]
  _QWORD v18[20]; // [rsp+20h] [rbp-A0h] BYREF

  v2 = 1;
  v3 = a2;
  v15 = v18;
  v17 = 16;
  v18[0] = a2;
  while ( 1 )
  {
    v16 = v2 - 1;
    v6 = sub_33CEEC0(a1, v3);
    if ( ((*(_BYTE *)(v3 + 32) & 4) != 0) != v6 )
      break;
    v2 = v16;
    v5 = v15;
    if ( !v16 )
      goto LABEL_13;
LABEL_3:
    v3 = v5[v2 - 1];
  }
  v8 = v16;
  v9 = v17;
  *(_BYTE *)(v3 + 32) = *(_BYTE *)(v3 + 32) & 0xFB | (4 * (v6 & 1));
  v10 = *(_QWORD *)(v3 + 56);
  v2 = v8;
  if ( v10 )
  {
    v11 = v10;
    v12 = 0;
    do
    {
      v11 = *(_QWORD *)(v11 + 32);
      ++v12;
    }
    while ( v11 );
    if ( v9 < v8 + v12 )
    {
      v14 = v12;
      sub_C8D5F0((__int64)&v15, v18, v8 + v12, 8u, v8, v7);
      LODWORD(v12) = v14;
      v13 = &v15[v16];
    }
    else
    {
      v13 = &v15[v8];
    }
    do
    {
      *v13++ = *(_QWORD *)(v10 + 16);
      v10 = *(_QWORD *)(v10 + 32);
    }
    while ( v10 );
    v2 = v16 + v12;
  }
  else if ( v8 > v9 )
  {
    sub_C8D5F0((__int64)&v15, v18, v8, 8u, v8, v7);
    v2 = v16;
  }
  v16 = v2;
  v5 = v15;
  if ( v2 )
    goto LABEL_3;
LABEL_13:
  if ( v5 != v18 )
    _libc_free((unsigned __int64)v5);
}
