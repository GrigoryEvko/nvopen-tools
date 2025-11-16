// Function: sub_67B420
// Address: 0x67b420
//
__int64 sub_67B420()
{
  int v0; // ebx
  __int64 v1; // rdx
  __int64 v2; // rcx
  unsigned int v3; // r12d
  _QWORD v5[3]; // [rsp+0h] [rbp-1D0h] BYREF
  int v6; // [rsp+18h] [rbp-1B8h]
  _BYTE v7[4]; // [rsp+1Ch] [rbp-1B4h] BYREF
  int v8; // [rsp+20h] [rbp-1B0h]
  _BYTE v9[4]; // [rsp+24h] [rbp-1ACh] BYREF
  _BYTE v10[4]; // [rsp+28h] [rbp-1A8h] BYREF
  int v11; // [rsp+2Ch] [rbp-1A4h]
  unsigned int v12; // [rsp+30h] [rbp-1A0h]
  _BYTE v13[8]; // [rsp+38h] [rbp-198h] BYREF
  _WORD v14[176]; // [rsp+40h] [rbp-190h] BYREF
  int v15; // [rsp+1A0h] [rbp-30h]
  __int16 v16; // [rsp+1A4h] [rbp-2Ch]

  v0 = 0;
  v5[1] = 0x100000000LL;
  v5[0] = 0;
  v5[2] = 0;
  v6 = 0;
  v12 = dword_4F06650[0];
  v8 = 1;
  v11 = 0;
  sub_7BDB60(1);
  sub_866940(0, v7, v13, v9, v10);
  memset(v14, 0, sizeof(v14));
  HIBYTE(v14[37]) = 1;
  v14[27] = 257;
  v15 = 0;
  v16 = 0;
  while ( 1 )
  {
    sub_7C6890(0, v14);
    v3 = word_4F06418[0];
    if ( word_4F06418[0] == 54 )
    {
      ++v0;
      goto LABEL_5;
    }
    if ( word_4F06418[0] != 55 )
      break;
    if ( !v0 )
      goto LABEL_10;
    --v0;
LABEL_5:
    sub_7B8B50(0, v14, v1, v2);
  }
  if ( word_4F06418[0] != 9 )
    v3 = v0 == 0 ? 0x4B : 0;
LABEL_10:
  sub_679880((__int64)v5);
  return v3;
}
