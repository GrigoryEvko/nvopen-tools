// Function: sub_1667B10
// Address: 0x1667b10
//
void __fastcall sub_1667B10(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rdi
  char v7; // cl
  __int64 v8; // rcx
  char v9; // si
  __int64 v10; // r8
  const char *v11; // rax
  __int64 v12; // r14
  _BYTE *v13; // rax
  __int64 v14; // rax
  const char *v15; // rax
  _QWORD v16[2]; // [rsp+0h] [rbp-40h] BYREF
  char v17; // [rsp+10h] [rbp-30h]
  char v18; // [rsp+11h] [rbp-2Fh]

  v4 = **(_QWORD **)(a2 - 24);
  v5 = *(_BYTE *)(v4 + 8);
  v6 = v4;
  v7 = v5;
  if ( v5 == 16 )
  {
    v6 = **(_QWORD **)(v4 + 16);
    v7 = *(_BYTE *)(v6 + 8);
  }
  if ( v7 == 15 )
  {
    v8 = *(_QWORD *)a2;
    v9 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
    v10 = *(_QWORD *)a2;
    if ( v9 == 16 )
    {
      v10 = **(_QWORD **)(v8 + 16);
      if ( *(_BYTE *)(v10 + 8) == 15 )
      {
LABEL_6:
        if ( *(_DWORD *)(v10 + 8) >> 8 == *(_DWORD *)(v6 + 8) >> 8 )
        {
          v18 = 1;
          v15 = "AddrSpaceCast must be between different address spaces";
        }
        else
        {
          if ( v5 != 16 || *(_DWORD *)(v8 + 32) == *(_DWORD *)(v4 + 32) )
          {
            sub_1663F80(a1, a2);
            return;
          }
          v18 = 1;
          v15 = "AddrSpaceCast vector pointer number of elements mismatch";
        }
        v16[0] = v15;
        v17 = 3;
        sub_164FF40((__int64 *)a1, (__int64)v16);
        if ( !*(_QWORD *)a1 )
          return;
LABEL_16:
        sub_164FA80((__int64 *)a1, a2);
        return;
      }
    }
    else if ( v9 == 15 )
    {
      goto LABEL_6;
    }
    v18 = 1;
    v11 = "AddrSpaceCast result must be a pointer";
  }
  else
  {
    v18 = 1;
    v11 = "AddrSpaceCast source must be a pointer";
  }
  v12 = *(_QWORD *)a1;
  v16[0] = v11;
  v17 = 3;
  if ( !v12 )
  {
    *(_BYTE *)(a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(v16, v12);
  v13 = *(_BYTE **)(v12 + 24);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 16) )
  {
    sub_16E7DE0(v12, 10);
  }
  else
  {
    *(_QWORD *)(v12 + 24) = v13 + 1;
    *v13 = 10;
  }
  v14 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 72) = 1;
  if ( v14 )
    goto LABEL_16;
}
