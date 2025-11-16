// Function: sub_164FD80
// Address: 0x164fd80
//
__int64 __fastcall sub_164FD80(__int64 **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 *v7; // r13
  __int64 *v8; // rbx
  __int64 v9; // r12
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 *v12; // r12
  __int64 *v13; // rbx
  __int64 v14; // r13
  _BYTE *v15; // rax
  bool v16; // zf
  _QWORD v17[2]; // [rsp+0h] [rbp-70h] BYREF
  const char *v18; // [rsp+10h] [rbp-60h] BYREF
  _QWORD *v19; // [rsp+18h] [rbp-58h]
  __int16 v20; // [rsp+20h] [rbp-50h]
  const char **v21; // [rsp+30h] [rbp-40h] BYREF
  const char *v22; // [rsp+38h] [rbp-38h]
  __int16 v23; // [rsp+40h] [rbp-30h]

  v4 = *a1;
  v17[0] = a2;
  v17[1] = a3;
  v5 = *v4;
  if ( a4 >= *(_DWORD *)(*v4 + 12) - 1 )
  {
    v12 = a1[2];
    v18 = "'allocsize' ";
    v13 = a1[1];
    v19 = v17;
    v21 = &v18;
    v20 = 1283;
    v22 = " argument is out of bounds";
    v23 = 770;
    v14 = *v12;
    if ( *v12 )
    {
      sub_16E2CE0(&v21, *v12);
      v15 = *(_BYTE **)(v14 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
      {
        sub_16E7DE0(v14, 10);
      }
      else
      {
        *(_QWORD *)(v14 + 24) = v15 + 1;
        *v15 = 10;
      }
    }
    v16 = *v12 == 0;
    *((_BYTE *)v12 + 72) = 1;
    if ( v16 )
      return 0;
    sub_164FA80(v12, *v13);
    return 0;
  }
  else
  {
    result = 1;
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v5 + 16) + 8LL * (a4 + 1)) + 8LL) == 11 )
      return result;
    v7 = a1[2];
    v8 = a1[1];
    v20 = 1283;
    v18 = "'allocsize' ";
    v19 = v17;
    v21 = &v18;
    v22 = " argument must refer to an integer parameter";
    v23 = 770;
    v9 = *v7;
    if ( *v7 )
    {
      sub_16E2CE0(&v21, *v7);
      v10 = *(_BYTE **)(v9 + 24);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
      {
        sub_16E7DE0(v9, 10);
      }
      else
      {
        *(_QWORD *)(v9 + 24) = v10 + 1;
        *v10 = 10;
      }
      v11 = *v7;
      *((_BYTE *)v7 + 72) = 1;
      if ( v11 )
        sub_164FA80(v7, *v8);
      return 0;
    }
    *((_BYTE *)v7 + 72) = 1;
    return 0;
  }
}
