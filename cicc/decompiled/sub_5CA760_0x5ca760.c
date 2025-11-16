// Function: sub_5CA760
// Address: 0x5ca760
//
__int64 __fastcall sub_5CA760(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r13
  __int64 v5; // rdi
  __int64 v8; // r14
  char *v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // r13
  bool v12; // r12
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+18h] [rbp-48h] BYREF
  _DWORD v18[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v4 = *(_QWORD *)(a1 + 32);
  v17 = a2;
  v5 = *(_QWORD *)(v4 + 40);
  if ( *(_BYTE *)(v5 + 173) != 12 )
  {
    v18[0] = 0;
    v8 = sub_620FA0(v5, v18);
    if ( v18[0] || (unsigned __int64)v8 > 0x7FFFFFFF )
    {
      v9 = sub_5C79F0(a1);
      sub_6851A0(1099, v4 + 24, v9);
      *(_BYTE *)(a1 + 8) = 0;
    }
    else
    {
      v10 = *(_QWORD *)(sub_5C7B50(a1, (__int64)&v17, a3) + 168);
      v16 = v10;
      if ( (*(_BYTE *)(v10 + 16) & 2) == 0 )
        goto LABEL_21;
      v11 = *(_QWORD **)v10;
      v12 = *(_QWORD *)(v10 + 40) != 0;
      if ( *(_QWORD *)v10 )
      {
        v13 = v12 + 1LL;
        do
        {
          v14 = v13;
          if ( v13 == v8 )
          {
            if ( !(unsigned int)sub_8D2E30(v11[1]) || (v15 = sub_8D46C0(v11[1]), !(unsigned int)sub_8D29E0(v15)) )
            {
              sub_6851C0(1138, a1 + 56);
              *(_BYTE *)(a1 + 8) = 0;
            }
          }
          v11 = (_QWORD *)*v11;
          ++v13;
        }
        while ( v11 );
      }
      else
      {
        v14 = v12;
      }
      if ( v8 > v14 )
      {
        sub_6851C0(1137, a1 + 56);
        *(_BYTE *)(a1 + 8) = 0;
      }
      else
      {
LABEL_21:
        if ( *(_BYTE *)(a1 + 8) )
          *(_DWORD *)(v16 + 28) = v8;
      }
    }
  }
  return v17;
}
