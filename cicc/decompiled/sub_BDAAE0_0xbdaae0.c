// Function: sub_BDAAE0
// Address: 0xbdaae0
//
void __fastcall sub_BDAAE0(__int64 a1, const char *a2)
{
  unsigned __int8 v4; // al
  const char **v5; // rdx
  const char *v6; // r13
  __int64 v7; // r14
  _BYTE *v8; // rax
  __int64 v9; // rsi
  char v10; // al
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax
  const char *v15; // [rsp-58h] [rbp-58h] BYREF
  char v16; // [rsp-38h] [rbp-38h]
  char v17; // [rsp-37h] [rbp-37h]

  if ( *a2 != 16 )
  {
    v4 = *(a2 - 16);
    v5 = (v4 & 2) != 0 ? (const char **)*((_QWORD *)a2 - 4) : (const char **)&a2[-8 * ((v4 >> 2) & 0xF) - 16];
    v6 = *v5;
    if ( *v5 )
    {
      if ( *v6 != 16 )
      {
        v7 = *(_QWORD *)a1;
        v17 = 1;
        v15 = "invalid file";
        v16 = 3;
        if ( v7 )
        {
          sub_CA0E80(&v15, v7);
          v8 = *(_BYTE **)(v7 + 32);
          if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 24) )
          {
            sub_CB5D20(v7, 10);
          }
          else
          {
            *(_QWORD *)(v7 + 32) = v8 + 1;
            *v8 = 10;
          }
          v9 = *(_QWORD *)a1;
          v10 = *(_BYTE *)(a1 + 154);
          *(_BYTE *)(a1 + 153) = 1;
          *(_BYTE *)(a1 + 152) |= v10;
          if ( v9 )
          {
            sub_A62C00(a2, v9, a1 + 16, *(_QWORD *)(a1 + 8));
            v11 = *(_QWORD *)a1;
            v12 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
            if ( (unsigned __int64)v12 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            {
              sub_CB5D20(v11, 10);
            }
            else
            {
              *(_QWORD *)(v11 + 32) = v12 + 1;
              *v12 = 10;
            }
            sub_A62C00(v6, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
            v13 = *(_QWORD *)a1;
            v14 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
            if ( (unsigned __int64)v14 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            {
              sub_CB5D20(v13, 10);
            }
            else
            {
              *(_QWORD *)(v13 + 32) = v14 + 1;
              *v14 = 10;
            }
          }
        }
        else
        {
          *(_BYTE *)(a1 + 153) = 1;
          *(_BYTE *)(a1 + 152) |= *(_BYTE *)(a1 + 154);
        }
      }
    }
  }
}
