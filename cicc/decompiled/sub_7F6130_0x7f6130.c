// Function: sub_7F6130
// Address: 0x7f6130
//
_QWORD *__fastcall sub_7F6130(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  unsigned int v4; // eax
  unsigned __int8 v5; // [rsp+Ch] [rbp-4h]

  v1 = sub_8D46C0(a1);
  v2 = sub_691620(v1);
  if ( *(char *)(v2 + 142) < 0 || *(_BYTE *)(v2 + 140) != 12 )
    return sub_73A8E0(*(unsigned int *)(v2 + 136), byte_4F06A51[0]);
  v5 = byte_4F06A51[0];
  v4 = sub_8D4AB0(v2, byte_4F06A51[0], byte_4F06A51);
  return sub_73A8E0(v4, v5);
}
