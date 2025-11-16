// Function: sub_19C8D10
// Address: 0x19c8d10
//
__int64 __fastcall sub_19C8D10(_QWORD *a1)
{
  __int64 v1; // rbx
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r15
  unsigned __int64 v4; // rdx
  _QWORD *v5; // r14
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v9; // rax
  _QWORD *v10; // rbx
  _QWORD *i; // r15
  __int64 v12; // rax
  _QWORD *v13; // [rsp+0h] [rbp-A0h]
  void *v14; // [rsp+10h] [rbp-90h] BYREF
  __int64 v15; // [rsp+18h] [rbp-88h] BYREF
  __int64 v16; // [rsp+28h] [rbp-78h]
  void *v17; // [rsp+40h] [rbp-60h] BYREF
  __int64 v18; // [rsp+48h] [rbp-58h] BYREF
  __int64 v19; // [rsp+58h] [rbp-48h]

  v1 = a1[25];
  *a1 = off_49F4790;
  v13 = (_QWORD *)v1;
  if ( v1 )
  {
    sub_1359CD0(v1);
    if ( *(_DWORD *)(v1 + 48) )
    {
      sub_1359800(&v14, -8, 0);
      sub_1359800(&v17, -16, 0);
      v9 = v1;
      v10 = *(_QWORD **)(v1 + 32);
      for ( i = &v10[6 * *(unsigned int *)(v9 + 48)]; i != v10; v10 += 6 )
      {
        v12 = v10[3];
        *v10 = &unk_49EE2B0;
        if ( v12 != 0 && v12 != -8 && v12 != -16 )
          sub_1649B30(v10 + 1);
      }
      v17 = &unk_49EE2B0;
      if ( v19 != 0 && v19 != -8 && v19 != -16 )
        sub_1649B30(&v18);
      v14 = &unk_49EE2B0;
      if ( v16 != -8 && v16 != 0 && v16 != -16 )
        sub_1649B30(&v15);
    }
    j___libc_free_0(v13[4]);
    v2 = (unsigned __int64 *)v13[2];
    while ( v13 + 1 != v2 )
    {
      v3 = v2;
      v2 = (unsigned __int64 *)v2[1];
      v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
      *v2 = v4 | *v2 & 7;
      *(_QWORD *)(v4 + 8) = v2;
      v5 = (_QWORD *)v3[6];
      v6 = (_QWORD *)v3[5];
      *v3 &= 7u;
      v3[1] = 0;
      if ( v5 != v6 )
      {
        do
        {
          v7 = v6[2];
          if ( v7 != -8 && v7 != 0 && v7 != -16 )
            sub_1649B30(v6);
          v6 += 3;
        }
        while ( v5 != v6 );
        v6 = (_QWORD *)v3[5];
      }
      if ( v6 )
        j_j___libc_free_0(v6, v3[7] - (_QWORD)v6);
      j_j___libc_free_0(v3, 72);
    }
    j_j___libc_free_0(v13, 72);
  }
  *a1 = &unk_49EAEF0;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 240);
}
