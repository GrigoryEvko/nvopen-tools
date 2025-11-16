// Function: sub_1D48EC0
// Address: 0x1d48ec0
//
void *__fastcall sub_1D48EC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // r13
  __int64 v8; // r13
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // r13
  __int64 v12; // r14
  unsigned __int64 v13; // rdi

  v7 = a1[35];
  *a1 = &unk_49F9B60;
  if ( v7 )
  {
    sub_1D489C0(v7);
    a2 = 776;
    j_j___libc_free_0(v7, 776);
  }
  v8 = a1[34];
  if ( v8 )
  {
    sub_1D17470(a1[34], a2, a3, a4, a5, a6);
    j_j___libc_free_0(v8, 904);
  }
  v9 = a1[31];
  if ( v9 )
  {
    sub_1D48CA0(a1[31]);
    j_j___libc_free_0(v9, 1008);
  }
  v10 = a1[53];
  if ( v10 )
    j_j___libc_free_0(v10, a1[55] - v10);
  v11 = a1[51];
  if ( v11 )
  {
    v12 = *(_QWORD *)(v11 + 16);
    if ( v12 )
    {
      sub_1368A00(*(__int64 **)(v11 + 16));
      j_j___libc_free_0(v12, 8);
    }
    j_j___libc_free_0(v11, 24);
  }
  v13 = a1[44];
  if ( v13 != a1[43] )
    _libc_free(v13);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
