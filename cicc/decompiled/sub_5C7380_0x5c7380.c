// Function: sub_5C7380
// Address: 0x5c7380
//
__int64 __fastcall sub_5C7380(__int64 a1, __int64 a2)
{
  bool v2; // r15
  __int64 v3; // r14
  char v4; // dl
  __int64 v5; // r13
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // r14
  int v11; // r13d
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  __int64 i; // rax
  int v16; // eax
  int v17; // eax
  int v18; // eax
  int v19; // eax
  int v20; // [rsp+Ch] [rbp-44h]
  _DWORD v21[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = 0;
  v21[0] = 0;
  v3 = *(_QWORD *)(a1 + 32);
  if ( unk_4F077B4 )
    v2 = unk_4F077A0 > 0x249EFu;
  v4 = *(_BYTE *)(a2 + 140);
  v5 = *(_QWORD *)(a1 + 48);
  if ( v4 == 12 )
  {
    v6 = a2;
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
      v4 = *(_BYTE *)(v6 + 140);
    }
    while ( v4 == 12 );
  }
  v7 = 1;
  if ( v4 )
  {
    if ( (unsigned int)sub_8D2AF0(a2) )
      goto LABEL_24;
    v16 = sub_8D2780(a2);
    v7 = 0;
    if ( !v16 || !v2 && (v17 = sub_8D29A0(a2), v7 = 0, v17) )
    {
      v18 = sub_8D2A90(a2);
      v7 = 0;
      if ( !v18 )
      {
        v19 = sub_8D3D40(a2);
        v7 = 0;
        if ( !v19 )
        {
LABEL_24:
          sub_6851C0(!v2 ? 3018 : 3442, a1 + 56);
          v7 = 1;
        }
      }
    }
  }
  v8 = *(_QWORD *)(v3 + 40);
  v9 = 1;
  if ( *(_BYTE *)(v8 + 173) != 12 )
  {
    v20 = v7;
    v9 = sub_620FA0(v8, v21);
    if ( v21[0] || (v7 = v20, (unsigned __int64)(v9 - 1) > 0x7FE) )
    {
      v14 = a1 + 56;
      sub_6851C0(3016, a1 + 56);
      if ( *(_BYTE *)(v5 + 268) == 4 || (*(_BYTE *)(v5 + 131) & 0x20) != 0 )
        goto LABEL_17;
      goto LABEL_16;
    }
  }
  if ( *(_BYTE *)(v5 + 268) != 4 && (*(_BYTE *)(v5 + 131) & 0x20) == 0 )
  {
    v14 = a1 + 56;
LABEL_16:
    sub_6851C0(3017, v14);
    goto LABEL_17;
  }
  if ( v7 )
  {
LABEL_17:
    *(_BYTE *)(a1 + 8) = 0;
    return sub_72C930();
  }
  if ( (unsigned int)sub_8D29A0(a2) )
  {
    if ( v9 <= 8 )
    {
      v11 = 1;
      v10 = 1;
    }
    else
    {
      v10 = ((unsigned int)((((((unsigned __int64)(((v9 + 7) >> 3) - 1) >> 1) | (((v9 + 7) >> 3) - 1)) >> 2)
                           | ((unsigned __int64)(((v9 + 7) >> 3) - 1) >> 1)
                           | (((v9 + 7) >> 3) - 1)) >> 4)
           | (unsigned int)((((unsigned __int64)(((v9 + 7) >> 3) - 1) >> 1) | (((v9 + 7) >> 3) - 1)) >> 2)
           | (unsigned int)((unsigned __int64)(((v9 + 7) >> 3) - 1) >> 1)
           | ((unsigned int)((v9 + 7) >> 3) - 1))
          + 1;
      v11 = v10;
    }
  }
  else
  {
    for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v10 = *(_QWORD *)(i + 128) * v9;
    v11 = v10;
  }
  result = sub_7259C0(15);
  v13 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(result + 128) = v10;
  *(_QWORD *)(result + 64) = v13;
  *(_DWORD *)(result + 136) = v11;
  *(_QWORD *)(result + 160) = a2;
  *(_QWORD *)(result + 168) = v8;
  *(_BYTE *)(result + 177) = 1;
  return result;
}
