// Function: sub_1E30990
// Address: 0x1e30990
//
__int64 __fastcall sub_1E30990(unsigned int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned int v5; // eax
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 result; // rax
  void *v9; // rdx
  _QWORD *v10; // rdx
  _BYTE v11[16]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]
  void (__fastcall *v13)(_BYTE *, __int64); // [rsp+18h] [rbp-28h]

  v3 = a2;
  if ( a3 )
  {
    v5 = sub_38D7150(a3 + 8, a1, 1);
    v6 = v5;
    if ( v5 == -1 )
    {
      v10 = *(_QWORD **)(v3 + 24);
      if ( *(_QWORD *)(v3 + 16) - (_QWORD)v10 <= 7u )
      {
        return sub_16E7EE0(v3, "<badreg>", 8u);
      }
      else
      {
        *v10 = 0x3E6765726461623CLL;
        *(_QWORD *)(v3 + 24) += 8LL;
        return 0x3E6765726461623CLL;
      }
    }
    else
    {
      sub_1F4AA00(v11, v5, a3, 0, 0);
      if ( !v12 )
        sub_4263D6(v11, v6, v7);
      v13(v11, v3);
      result = (__int64)v12;
      if ( v12 )
        return v12(v11, v11, 3);
    }
  }
  else
  {
    v9 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v9 <= 9u )
    {
      v3 = sub_16E7EE0(a2, "%dwarfreg.", 0xAu);
    }
    else
    {
      qmemcpy(v9, "%dwarfreg.", 10);
      *(_QWORD *)(a2 + 24) += 10LL;
    }
    return sub_16E7A90(v3, a1);
  }
  return result;
}
