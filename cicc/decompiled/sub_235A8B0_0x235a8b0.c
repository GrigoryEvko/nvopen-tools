// Function: sub_235A8B0
// Address: 0x235a8b0
//
unsigned __int64 __fastcall sub_235A8B0(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int16 v3; // bx
  __int64 v4; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *a2;
  *a2 = 0;
  v3 = *((_WORD *)a2 + 4);
  v4 = sub_22077B0(0x18u);
  if ( v4 )
  {
    *(_QWORD *)(v4 + 8) = v2;
    *(_WORD *)(v4 + 16) = v3;
    v6[0] = v4;
    *(_QWORD *)v4 = &unk_4A12538;
    result = sub_235A870(a1, v6);
    if ( v6[0] )
      return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v6[0] + 8LL))(v6[0]);
  }
  else
  {
    v6[0] = 0;
    result = sub_235A870(a1, v6);
    if ( v6[0] )
      result = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v6[0] + 8LL))(v6[0]);
    if ( v2 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  }
  return result;
}
