// Function: sub_164F0A0
// Address: 0x164f0a0
//
void __fastcall sub_164F0A0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r13
  __int64 v4; // r14
  _BYTE *v6; // rax
  __int64 v7; // rsi
  char v8; // al
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rdi
  _BYTE *v12; // rax
  const char *v13; // [rsp-48h] [rbp-48h] BYREF
  char v14; // [rsp-38h] [rbp-38h]
  char v15; // [rsp-37h] [rbp-37h]

  if ( *(_BYTE *)a2 != 15 )
  {
    v3 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
    if ( v3 )
    {
      if ( *v3 != 15 )
      {
        v4 = *(_QWORD *)a1;
        v15 = 1;
        v13 = "invalid file";
        v14 = 3;
        if ( v4 )
        {
          sub_16E2CE0(&v13, v4);
          v6 = *(_BYTE **)(v4 + 24);
          if ( (unsigned __int64)v6 >= *(_QWORD *)(v4 + 16) )
          {
            sub_16E7DE0(v4, 10);
          }
          else
          {
            *(_QWORD *)(v4 + 24) = v6 + 1;
            *v6 = 10;
          }
          v7 = *(_QWORD *)a1;
          v8 = *(_BYTE *)(a1 + 74);
          *(_BYTE *)(a1 + 73) = 1;
          *(_BYTE *)(a1 + 72) |= v8;
          if ( v7 )
          {
            sub_15562E0((unsigned __int8 *)a2, v7, a1 + 16, *(_QWORD *)(a1 + 8));
            v9 = *(_QWORD *)a1;
            v10 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
            if ( (unsigned __int64)v10 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            {
              sub_16E7DE0(v9, 10);
            }
            else
            {
              *(_QWORD *)(v9 + 24) = v10 + 1;
              *v10 = 10;
            }
            sub_15562E0(v3, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
            v11 = *(_QWORD *)a1;
            v12 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
            if ( (unsigned __int64)v12 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            {
              sub_16E7DE0(v11, 10);
            }
            else
            {
              *(_QWORD *)(v11 + 24) = v12 + 1;
              *v12 = 10;
            }
          }
        }
        else
        {
          *(_BYTE *)(a1 + 73) = 1;
          *(_BYTE *)(a1 + 72) |= *(_BYTE *)(a1 + 74);
        }
      }
    }
  }
}
