// Function: sub_38F0890
// Address: 0x38f0890
//
__int64 __fastcall sub_38F0890(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  const char *v14; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15; // [rsp+18h] [rbp-38h]
  _QWORD v16[2]; // [rsp+20h] [rbp-30h] BYREF
  __int16 v17; // [rsp+30h] [rbp-20h]

  v6 = *(_QWORD *)(a1 + 400);
  if ( *(_QWORD *)(a1 + 392) != v6 && *(_BYTE *)(v6 - 3) )
  {
    sub_38F0630(a1);
    return 0;
  }
  else
  {
    if ( a3 )
    {
      v15 = 39;
      v14 = ".error directive invoked in source file";
      v7 = **(_DWORD **)(a1 + 152);
      if ( v7 != 9 )
      {
        if ( v7 != 3 )
        {
          v16[0] = ".error argument must be a string";
          v17 = 259;
          return sub_3909CF0(a1, v16, 0, 0, a5, a6);
        }
        v9 = sub_3909460(a1);
        v10 = 0;
        v11 = v9;
        v12 = *(_QWORD *)(v9 + 16);
        if ( v12 )
        {
          v13 = v12 - 1;
          if ( v12 == 1 )
            v13 = 1;
          if ( v13 > v12 )
            v13 = v12;
          v12 = 1;
          v10 = v13 - 1;
        }
        v14 = (const char *)(*(_QWORD *)(v11 + 8) + v12);
        v15 = v10;
        sub_38EB180(a1);
      }
      v17 = 261;
      v16[0] = &v14;
    }
    else
    {
      v16[0] = ".err encountered";
      v17 = 259;
    }
    return sub_3909790(a1, a2, v16, 0, 0);
  }
}
