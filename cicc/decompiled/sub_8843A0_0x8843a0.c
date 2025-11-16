// Function: sub_8843A0
// Address: 0x8843a0
//
__int64 __fastcall sub_8843A0(__int64 a1, __int64 a2, FILE *a3, __int64 a4, _DWORD *a5)
{
  __int64 v6; // r13
  char v9; // al
  int v10; // eax
  __int64 result; // rax
  char v12; // cl
  __int64 v13; // rax
  char v14; // dl
  char v15; // al
  bool v16; // dl
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned int v19; // [rsp+8h] [rbp-48h]
  unsigned __int8 v20[49]; // [rsp+1Fh] [rbp-31h] BYREF

  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
    return 1;
  v6 = a1;
  if ( qword_4D0495C )
  {
    v9 = *(_BYTE *)(a1 + 80);
    if ( v9 == 16 )
    {
      a1 = **(_QWORD **)(a1 + 88);
      v9 = *(_BYTE *)(a1 + 80);
    }
    if ( v9 == 24 )
      a1 = *(_QWORD *)(a1 + 88);
    v10 = (unsigned __int8)sub_87D550(a1);
  }
  else
  {
    sub_883A10(a1, a2, v20);
    v10 = v20[0];
  }
  if ( v10 != 1 )
    return 1;
  if ( a4 )
  {
    v12 = *(_BYTE *)(a4 + 140);
    if ( v12 == 12 )
    {
      v13 = a4;
      do
      {
        v13 = *(_QWORD *)(v13 + 160);
        v14 = *(_BYTE *)(v13 + 140);
      }
      while ( v14 == 12 );
    }
    else
    {
      v14 = *(_BYTE *)(a4 + 140);
    }
    if ( !v14 )
      return 1;
    v15 = *(_BYTE *)(v6 + 80);
    if ( v15 == 16 )
    {
      v6 = **(_QWORD **)(v6 + 88);
      v15 = *(_BYTE *)(v6 + 80);
    }
    if ( v15 == 24 )
      v6 = *(_QWORD *)(v6 + 88);
    if ( v12 == 12 )
    {
      do
        a4 = *(_QWORD *)(a4 + 160);
      while ( *(_BYTE *)(a4 + 140) == 12 );
    }
    v18 = *(_QWORD *)(v6 + 64);
    result = sub_87D890(a4);
    if ( (_DWORD)result )
      return 1;
    v16 = 1;
    if ( v18 != a4
      && (!v18 || !dword_4F07588 || !(v16 = *(_QWORD *)(a4 + 32) != 0 && *(_QWORD *)(v18 + 32) == *(_QWORD *)(a4 + 32))) )
    {
      v17 = sub_8D5CE0(a4, v18);
      result = sub_87D8D0(*(_QWORD **)(v17 + 112));
      v16 = (_DWORD)result == 0;
    }
  }
  else
  {
    v16 = 1;
    result = 0;
  }
  if ( a3 )
  {
    if ( v16 )
    {
      v19 = result;
      sub_87D9B0(v6, a2, a4, a3, 0, 3, 0, a5);
      return v19;
    }
  }
  return result;
}
