// Function: sub_17ED0F0
// Address: 0x17ed0f0
//
__int64 __fastcall sub_17ED0F0(unsigned __int64 a1, __int64 *a2)
{
  char v2; // r8
  __int64 result; // rax
  __int64 *v4; // r15
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rbx
  _QWORD *v11; // rdi
  _QWORD *v12; // rdi
  size_t v13; // rdx
  void *s; // [rsp+10h] [rbp-70h] BYREF
  __int64 v15; // [rsp+18h] [rbp-68h]
  _QWORD *v16; // [rsp+20h] [rbp-60h]
  __int64 v17; // [rsp+28h] [rbp-58h]
  int v18; // [rsp+30h] [rbp-50h]
  __int64 v19; // [rsp+38h] [rbp-48h]
  _QWORD v20[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = sub_1636800(a1, a2);
  result = 0;
  if ( !v2 )
  {
    v4 = (__int64 *)a2[4];
    v15 = 1;
    s = v20;
    v16 = 0;
    v17 = 0;
    v18 = 1065353216;
    v19 = 0;
    v20[0] = 0;
    if ( v4 == a2 + 3 )
    {
      v12 = v20;
      v13 = 8;
    }
    else
    {
      do
      {
        v5 = 0;
        if ( v4 )
          v5 = v4 - 7;
        if ( !sub_15E4F60((__int64)v5) )
        {
          v6 = sub_161ACC0(*(_QWORD *)(a1 + 8), a1, (__int64)&unk_4F98724, (unsigned __int64)v5);
          v7 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v6 + 104LL))(v6, &unk_4F98724);
          v8 = sub_161ACC0(*(_QWORD *)(a1 + 8), a1, (__int64)&unk_4F97E48, (unsigned __int64)v5);
          v9 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_4F97E48);
          sub_17EB290(v5, a2, v7 + 160, (__int64 *)(v9 + 160), &s);
        }
        v4 = (__int64 *)v4[1];
      }
      while ( a2 + 3 != v4 );
      v10 = v16;
      while ( v10 )
      {
        v11 = v10;
        v10 = (_QWORD *)*v10;
        j_j___libc_free_0(v11, 24);
      }
      v12 = s;
      v13 = 8 * v15;
    }
    memset(v12, 0, v13);
    v17 = 0;
    v16 = 0;
    if ( s != v20 )
      j_j___libc_free_0(s, 8 * v15);
    return 1;
  }
  return result;
}
