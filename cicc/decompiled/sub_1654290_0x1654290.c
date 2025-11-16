// Function: sub_1654290
// Address: 0x1654290
//
void __fastcall sub_1654290(__int64 *a1, __int64 a2)
{
  __int64 v4; // r14
  _BYTE *v5; // rax
  __int64 v6; // rsi
  char v7; // al
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rbx
  char *i; // r15
  const char *v16; // rax
  __int64 v17; // [rsp+8h] [rbp-58h]
  _QWORD v18[2]; // [rsp+10h] [rbp-50h] BYREF
  char v19; // [rsp+20h] [rbp-40h]
  char v20; // [rsp+21h] [rbp-3Fh]

  if ( *(_WORD *)(a2 + 2) != 41 )
  {
    v4 = *a1;
    v20 = 1;
    v18[0] = "invalid tag";
    v19 = 3;
    if ( v4 )
    {
      sub_16E2CE0(v18, v4);
      v5 = *(_BYTE **)(v4 + 24);
      if ( (unsigned __int64)v5 >= *(_QWORD *)(v4 + 16) )
      {
        sub_16E7DE0(v4, 10);
      }
      else
      {
        *(_QWORD *)(v4 + 24) = v5 + 1;
        *v5 = 10;
      }
      v6 = *a1;
      v7 = *((_BYTE *)a1 + 74);
      *((_BYTE *)a1 + 73) = 1;
      *((_BYTE *)a1 + 72) |= v7;
      if ( v6 )
      {
        sub_15562E0((unsigned __int8 *)a2, v6, (__int64)(a1 + 2), a1[1]);
        v8 = *a1;
        v9 = *(_BYTE **)(*a1 + 24);
        if ( (unsigned __int64)v9 >= *(_QWORD *)(*a1 + 16) )
        {
          sub_16E7DE0(v8, 10);
        }
        else
        {
          *(_QWORD *)(v8 + 24) = v9 + 1;
          *v9 = 10;
        }
      }
    }
    else
    {
      *((_BYTE *)a1 + 73) = 1;
      *((_BYTE *)a1 + 72) |= *((_BYTE *)a1 + 74);
    }
    return;
  }
  if ( !*(_BYTE *)(a2 + 40) )
    return;
  v10 = sub_161E970(*(_QWORD *)(a2 + 32));
  v17 = v11;
  v12 = *(_DWORD *)(a2 + 24);
  if ( v12 > 2 )
  {
    v20 = 1;
    v16 = "invalid checksum kind";
    goto LABEL_22;
  }
  v13 = 32;
  if ( v12 == 2 )
    v13 = 40;
  if ( v13 != v17 )
  {
    v20 = 1;
    v16 = "invalid checksum length";
LABEL_22:
    v18[0] = v16;
    v19 = 3;
    sub_16521E0(a1, (__int64)v18);
    if ( *a1 )
      sub_164ED40(a1, (unsigned __int8 *)a2);
    return;
  }
  v14 = v17;
  for ( i = (char *)v10; (unsigned __int8)sub_164DB70((__int64 (__fastcall *)(_QWORD))sub_164DB40, *i); ++i )
  {
    if ( !--v14 )
      return;
  }
  if ( v17 - v14 != -1 )
  {
    v20 = 1;
    v16 = "invalid checksum";
    goto LABEL_22;
  }
}
