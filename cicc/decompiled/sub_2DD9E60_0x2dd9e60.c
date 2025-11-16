// Function: sub_2DD9E60
// Address: 0x2dd9e60
//
char __fastcall sub_2DD9E60(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 *v6; // rsi
  _BYTE *v7; // rax
  _BYTE *v8; // r12
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 i; // r14
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = (__int64 *)a3;
  v7 = sub_BA8CD0(a2, a3, a4, 0);
  if ( v7 )
  {
    v8 = v7;
    LOBYTE(v7) = sub_B2FC80((__int64)v7);
    if ( !(_BYTE)v7 )
    {
      v9 = *((_QWORD *)v8 - 4);
      LODWORD(v7) = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
      if ( (_DWORD)v7 )
      {
        v10 = (unsigned int)((_DWORD)v7 - 1);
        for ( i = 0; ; ++i )
        {
          v7 = sub_BD3990(*(unsigned __int8 **)(v9 + 32 * (i - (unsigned int)v7)), (__int64)v6);
          if ( *v7 == 3 )
          {
            v6 = v17;
            v17[0] = (__int64)v7;
            LOBYTE(v7) = sub_2DD9820(a1 + 32, v17, v12, v13, v14, v15);
          }
          if ( i == v10 )
            break;
          LODWORD(v7) = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
        }
      }
    }
  }
  return (char)v7;
}
