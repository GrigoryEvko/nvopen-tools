// Function: sub_2EAAF50
// Address: 0x2eaaf50
//
__int64 __fastcall sub_2EAAF50(unsigned int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rdx
  __int64 result; // rax
  _QWORD *v7; // rdx
  void *v8; // rdx
  __int64 v9; // [rsp+8h] [rbp-48h]
  _BYTE v10[16]; // [rsp+10h] [rbp-40h] BYREF
  __int64 (__fastcall *v11)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-30h]
  void (__fastcall *v12)(_BYTE *, __int64); // [rsp+28h] [rbp-28h]

  v3 = a2;
  if ( a3 )
  {
    v9 = sub_E92200(a3, a1, 1);
    if ( BYTE4(v9) )
    {
      sub_2FF6320(v10, (unsigned int)v9, a3, 0, 0);
      if ( !v11 )
        sub_4263D6(v10, (unsigned int)v9, v5);
      v12(v10, a2);
      result = (__int64)v11;
      if ( v11 )
        return v11(v10, v10, 3);
    }
    else
    {
      v7 = *(_QWORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 7u )
      {
        return sub_CB6200(a2, "<badreg>", 8u);
      }
      else
      {
        *v7 = 0x3E6765726461623CLL;
        *(_QWORD *)(a2 + 32) += 8LL;
        return 0x3E6765726461623CLL;
      }
    }
  }
  else
  {
    v8 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 <= 9u )
    {
      v3 = sub_CB6200(a2, "%dwarfreg.", 0xAu);
    }
    else
    {
      qmemcpy(v8, "%dwarfreg.", 10);
      *(_QWORD *)(a2 + 32) += 10LL;
    }
    return sub_CB59D0(v3, a1);
  }
  return result;
}
