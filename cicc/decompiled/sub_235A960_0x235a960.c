// Function: sub_235A960
// Address: 0x235a960
//
unsigned __int64 __fastcall sub_235A960(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // r12
  char v3; // r14
  int v4; // r13d
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 result; // rax
  __int64 v8; // [rsp+8h] [rbp-48h]
  unsigned __int64 v9[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *a2;
  *a2 = 0;
  v3 = *((_BYTE *)a2 + 8);
  v4 = *((_DWORD *)a2 + 3);
  v5 = a2[3];
  v8 = a2[2];
  v6 = sub_22077B0(0x28u);
  if ( v6 )
  {
    *(_QWORD *)(v6 + 8) = v2;
    *(_BYTE *)(v6 + 16) = v3;
    *(_DWORD *)(v6 + 20) = v4;
    *(_QWORD *)(v6 + 24) = v8;
    *(_QWORD *)v6 = &unk_4A0ECB8;
    *(_QWORD *)(v6 + 32) = v5;
    v9[0] = v6;
    result = sub_235A870(a1, v9);
    if ( v9[0] )
      return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v9[0] + 8LL))(v9[0]);
  }
  else
  {
    v9[0] = 0;
    result = sub_235A870(a1, v9);
    if ( v9[0] )
      result = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v9[0] + 8LL))(v9[0]);
    if ( v2 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  }
  return result;
}
