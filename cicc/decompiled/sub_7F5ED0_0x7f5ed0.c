// Function: sub_7F5ED0
// Address: 0x7f5ed0
//
_QWORD *__fastcall sub_7F5ED0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi
  char v3; // al
  __int64 v5; // rax
  unsigned __int8 v6; // [rsp+Ch] [rbp-4h]

  v1 = sub_8D46C0(a1);
  v2 = sub_691620(v1);
  v3 = *(_BYTE *)(v2 + 140);
  if ( v3 == 12 )
  {
    v6 = byte_4F06A51[0];
    v5 = sub_8D4A00(v2);
    return sub_73A8E0(v5, v6);
  }
  else if ( dword_4F077C0 && (v3 == 1 || v3 == 7) )
  {
    return sub_73A8E0(1, byte_4F06A51[0]);
  }
  else
  {
    return sub_73A8E0(*(_QWORD *)(v2 + 128), byte_4F06A51[0]);
  }
}
