// Function: sub_2562110
// Address: 0x2562110
//
__int64 __fastcall sub_2562110(__int64 a1)
{
  __int64 v2; // rdi
  int v3; // r12d
  int v4; // ebx
  int v5; // eax
  _BYTE *v6; // r14
  int v7; // r13d
  int v8; // r13d
  unsigned int v9; // edx
  int v11; // [rsp+Ch] [rbp-44h]
  _BYTE v12[49]; // [rsp+1Fh] [rbp-31h] BYREF

  v2 = a1 + 72;
  v3 = 8;
  v4 = 1;
  v5 = *(_DWORD *)(v2 + 24);
  v12[0] = 0;
  v11 = v5;
  v6 = (_BYTE *)sub_250D070((_QWORD *)v2);
  if ( *v6 <= 0x1Cu )
    v6 = 0;
  do
  {
    while ( (v4 & v11) != 0 )
    {
      v4 *= 2;
      if ( !--v3 )
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
    v9 = v4;
    v4 *= 2;
    sub_2561E50(a1, a1 + 88, v9, (__int64)v6, 0, (__int64)v12, v8);
    --v3;
  }
  while ( v3 );
LABEL_9:
  *(_DWORD *)(a1 + 100) = *(_DWORD *)(a1 + 96);
  return 0;
}
