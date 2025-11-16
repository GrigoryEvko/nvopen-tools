// Function: sub_ED7D90
// Address: 0xed7d90
//
__int64 *__fastcall sub_ED7D90(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v7; // rdx
  __int64 *result; // rax
  __int64 v9; // rdx
  char *v10; // rbx
  char *v11; // r14
  char v12; // al
  _BYTE *v13; // rax
  __int64 *v14; // [rsp+0h] [rbp-60h]
  __int64 *v15; // [rsp+8h] [rbp-58h]
  _QWORD v16[2]; // [rsp+10h] [rbp-50h] BYREF
  char v17; // [rsp+20h] [rbp-40h]

  v7 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v7 <= 0xCu )
  {
    sub_CB6200(a1, "Binary IDs: \n", 0xDu);
  }
  else
  {
    qmemcpy(v7, "Binary IDs: \n", 13);
    *(_QWORD *)(a1 + 32) += 13LL;
  }
  result = &a2[5 * a3];
  v14 = result;
  if ( result != a2 )
  {
    v15 = a2;
    do
    {
      v9 = *v15;
      v10 = (char *)(*v15 + v15[1]);
      if ( v10 != (char *)*v15 )
      {
        v11 = (char *)*v15;
        do
        {
          v12 = *v11++;
          v16[0] = &unk_49DD0D8;
          v16[1] = "%02x";
          v17 = v12;
          sub_CB6620(a1, (__int64)v16, v9, (__int64)"%02x", a5, a6);
        }
        while ( v10 != v11 );
      }
      v13 = *(_BYTE **)(a1 + 32);
      if ( *(_BYTE **)(a1 + 24) == v13 )
      {
        sub_CB6200(a1, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v13 = 10;
        ++*(_QWORD *)(a1 + 32);
      }
      v15 += 5;
      result = v15;
    }
    while ( v14 != v15 );
  }
  return result;
}
