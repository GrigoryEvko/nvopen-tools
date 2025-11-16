// Function: sub_19C8AD0
// Address: 0x19c8ad0
//
void *__fastcall sub_19C8AD0(_QWORD *a1)
{
  __int64 v1; // r14
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r15
  unsigned __int64 v4; // rdx
  _QWORD *v5; // r13
  _QWORD *v6; // r12
  __int64 v7; // rax
  _QWORD *v9; // rbx
  _QWORD *i; // r15
  __int64 v11; // rax
  void *v12; // [rsp+10h] [rbp-90h] BYREF
  __int64 v13; // [rsp+18h] [rbp-88h] BYREF
  __int64 v14; // [rsp+28h] [rbp-78h]
  void *v15; // [rsp+40h] [rbp-60h] BYREF
  __int64 v16; // [rsp+48h] [rbp-58h] BYREF
  __int64 v17; // [rsp+58h] [rbp-48h]

  v1 = a1[25];
  *a1 = off_49F4790;
  if ( v1 )
  {
    sub_1359CD0(v1);
    if ( *(_DWORD *)(v1 + 48) )
    {
      sub_1359800(&v12, -8, 0);
      sub_1359800(&v15, -16, 0);
      v9 = *(_QWORD **)(v1 + 32);
      for ( i = &v9[6 * *(unsigned int *)(v1 + 48)]; i != v9; v9 += 6 )
      {
        v11 = v9[3];
        *v9 = &unk_49EE2B0;
        if ( v11 != 0 && v11 != -8 && v11 != -16 )
          sub_1649B30(v9 + 1);
      }
      v15 = &unk_49EE2B0;
      if ( v17 != 0 && v17 != -8 && v17 != -16 )
        sub_1649B30(&v16);
      v12 = &unk_49EE2B0;
      if ( v14 != -8 && v14 != 0 && v14 != -16 )
        sub_1649B30(&v13);
    }
    j___libc_free_0(*(_QWORD *)(v1 + 32));
    v2 = *(unsigned __int64 **)(v1 + 16);
    while ( (unsigned __int64 *)(v1 + 8) != v2 )
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
    j_j___libc_free_0(v1, 72);
  }
  *a1 = &unk_49EAEF0;
  return sub_16366C0(a1);
}
