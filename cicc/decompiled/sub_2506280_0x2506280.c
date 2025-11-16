// Function: sub_2506280
// Address: 0x2506280
//
__int64 __fastcall sub_2506280(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 **a5)
{
  int v7; // ebx
  unsigned int v8; // eax
  unsigned int v9; // r12d
  int v10; // r13d
  unsigned int v12; // eax
  int v13; // esi
  int v14; // r15d
  size_t v15; // rdx
  size_t v16; // r15
  unsigned int v17; // eax
  const void *v18; // rax
  size_t v19; // rdx
  int v20; // r12d
  unsigned int v21; // r12d
  unsigned int v22; // esi
  unsigned __int64 v23; // rbx
  const void *v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = **(unsigned __int8 **)(a1 + 8);
  v25 = a3;
  LOBYTE(v8) = sub_A71800((__int64)a2);
  if ( (_BYTE)v8 )
  {
    v9 = v8;
    v10 = sub_A71AE0(a2);
    if ( !(unsigned __int8)sub_A73170(&v25, v10) )
    {
      sub_A77B20(a5, v10);
      return v9;
    }
    return 0;
  }
  if ( !sub_A71840((__int64)a2) )
  {
    LOBYTE(v12) = sub_A71820((__int64)a2);
    v9 = v12;
    if ( !(_BYTE)v12 )
      BUG();
    v14 = v7 ^ 1;
    v13 = sub_A71AE0(a2);
    LOBYTE(v14) = (v13 == 92) & (v7 ^ 1);
    if ( (_BYTE)v14 )
    {
      v20 = sub_A73AB0(&v25);
      v21 = sub_A71E40(a2) & v20;
      if ( v21 != (unsigned int)sub_A73AB0(&v25) )
      {
        v22 = v21;
        v9 = v14;
        sub_A77CD0(a5, v22);
        return v9;
      }
    }
    else if ( !(unsigned __int8)sub_A73170(&v25, v13)
           || (_BYTE)v7
           || (v26[0] = sub_A734C0(&v25, v13), sub_A71820((__int64)v26))
           && (v23 = sub_A71B80(v26), v23 < sub_A71B80(a2)) )
    {
      sub_A77670((__int64)a5, *a2);
      return v9;
    }
    return 0;
  }
  v24 = (const void *)sub_A71FD0(a2);
  v16 = v15;
  v17 = sub_A73380(&v25, v24, v15) ^ 1;
  v9 = v17;
  LOBYTE(v9) = v7 | v17;
  if ( (unsigned __int8)v7 | (unsigned __int8)v17 )
  {
    v18 = (const void *)sub_A72240(a2);
    sub_A78980(a5, v24, v16, v18, v19);
  }
  return v9;
}
