// Function: sub_6ED310
// Address: 0x6ed310
//
__int64 __fastcall sub_6ED310(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // [rsp-40h] [rbp-40h]
  __int64 v17; // [rsp-30h] [rbp-30h] BYREF

  result = 0;
  if ( *((_BYTE *)a1 + 24) == 1 && *((_BYTE *)a1 + 56) == 14 )
  {
    v8 = a1[9];
    if ( *(_BYTE *)(v8 + 24) == 2 )
    {
      v9 = *(_QWORD *)(v8 + 56);
      if ( *(_BYTE *)(v9 + 173) == 10 )
      {
        v11 = sub_724DC0(a1, a2, v9, a4, a5, a6);
        v12 = *a1;
        v17 = v11;
        v13 = sub_8D5CE0(*(_QWORD *)v8, v12);
        sub_72D460(*(_QWORD *)(v8 + 56), v17);
        v14 = v17;
        *(_QWORD *)(v17 + 192) = *(_QWORD *)(v13 + 104);
        v15 = v17;
        *(_QWORD *)(v14 + 128) = sub_72D2E0(*a1, 0);
        v16 = sub_718E10(v15, a2);
        sub_724E30(&v17);
        return v16;
      }
    }
  }
  return result;
}
