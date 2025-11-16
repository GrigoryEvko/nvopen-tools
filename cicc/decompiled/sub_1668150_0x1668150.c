// Function: sub_1668150
// Address: 0x1668150
//
void __fastcall sub_1668150(__int64 a1, _QWORD *a2)
{
  const char *v2; // rax
  __int64 v3; // r14
  _BYTE *v4; // rax
  __int64 v5; // rax
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+10h] [rbp-30h]
  char v8; // [rsp+11h] [rbp-2Fh]

  if ( sub_15F50B0((__int64 *)*(a2 - 9), *(a2 - 6), (_QWORD *)*(a2 - 3)) )
  {
    v8 = 1;
    v2 = "Invalid operands for select instruction!";
  }
  else
  {
    if ( *a2 == *(_QWORD *)*(a2 - 6) )
    {
      sub_1663F80(a1, (__int64)a2);
      return;
    }
    v8 = 1;
    v2 = "Select values must have same type as select instruction!";
  }
  v3 = *(_QWORD *)a1;
  v6 = v2;
  v7 = 3;
  if ( v3 )
  {
    sub_16E2CE0(&v6, v3);
    v4 = *(_BYTE **)(v3 + 24);
    if ( (unsigned __int64)v4 >= *(_QWORD *)(v3 + 16) )
    {
      sub_16E7DE0(v3, 10);
    }
    else
    {
      *(_QWORD *)(v3 + 24) = v4 + 1;
      *v4 = 10;
    }
    v5 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 72) = 1;
    if ( v5 )
      sub_164FA80((__int64 *)a1, (__int64)a2);
  }
  else
  {
    *(_BYTE *)(a1 + 72) = 1;
  }
}
