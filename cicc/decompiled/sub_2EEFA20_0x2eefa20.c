// Function: sub_2EEFA20
// Address: 0x2eefa20
//
__int64 __fastcall sub_2EEFA20(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  void *v3; // rdx
  __int64 v4; // rdx
  _BYTE *v5; // rax
  __int64 result; // rax
  _BYTE v7[16]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v8)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]
  void (__fastcall *v9)(_BYTE *, __int64); // [rsp+18h] [rbp-28h]

  if ( (a2 & 0x80000000) != 0 )
    return sub_2EEF700(a1, a2);
  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(void **)(v2 + 32);
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v3 <= 0xEu )
  {
    v2 = sub_CB6200(*(_QWORD *)(a1 + 16), "- regunit:     ", 0xFu);
  }
  else
  {
    qmemcpy(v3, "- regunit:     ", 15);
    *(_QWORD *)(v2 + 32) += 15LL;
  }
  sub_2FF6390(v7, a2, *(_QWORD *)(a1 + 56));
  if ( !v8 )
    sub_4263D6(v7, a2, v4);
  v9(v7, v2);
  v5 = *(_BYTE **)(v2 + 32);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(v2 + 24) )
  {
    sub_CB5D20(v2, 10);
  }
  else
  {
    *(_QWORD *)(v2 + 32) = v5 + 1;
    *v5 = 10;
  }
  result = (__int64)v8;
  if ( v8 )
    return v8(v7, v7, 3);
  return result;
}
