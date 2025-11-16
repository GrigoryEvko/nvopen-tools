// Function: sub_2562300
// Address: 0x2562300
//
__int64 __fastcall sub_2562300(__int64 a1)
{
  __int64 v1; // rax
  int v2; // r15d
  int v3; // r14d
  __int64 v5; // rdi
  _BYTE *v6; // r13
  int v7; // ebx
  int v8; // ebx
  unsigned int v9; // edx
  __int64 v11; // [rsp+10h] [rbp-50h]
  int v12; // [rsp+1Ch] [rbp-44h]
  _BYTE v13[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v1 = a1 - 88;
  v2 = 8;
  v3 = 1;
  v5 = a1 - 16;
  v11 = v1;
  LODWORD(v1) = *(_DWORD *)(v5 + 24);
  v13[0] = 0;
  v12 = v1;
  v6 = (_BYTE *)sub_250D070((_QWORD *)v5);
  if ( *v6 <= 0x1Cu )
    v6 = 0;
  do
  {
    while ( (v3 & v12) != 0 )
    {
      v3 *= 2;
      if ( !--v2 )
        goto LABEL_9;
    }
    if ( v6 )
    {
      v7 = (unsigned __int8)sub_B46420((__int64)v6);
      v8 = (2 * ((unsigned __int8)sub_B46490((__int64)v6) != 0)) | v7;
    }
    else
    {
      v8 = 3;
    }
    v9 = v3;
    v3 *= 2;
    sub_2561E50(v11, a1, v9, (__int64)v6, 0, (__int64)v13, v8);
    --v2;
  }
  while ( v2 );
LABEL_9:
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a1 + 8);
  return 0;
}
