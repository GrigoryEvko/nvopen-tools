// Function: sub_6F8FA0
// Address: 0x6f8fa0
//
__int64 *__fastcall sub_6F8FA0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 *result; // rax
  _DWORD v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = a1;
  v3 = sub_736020(*(_QWORD *)(a1 + 128), 1);
  *(_BYTE *)(v3 + 174) |= 8u;
  v4 = v3;
  result = (__int64 *)sub_6F8E70(v3, &dword_4F063F8, &qword_4F063F0, a2, 0);
  *(_BYTE *)(v4 + 177) = 1;
  if ( (*(_BYTE *)(a1 - 8) & 1) == 0 )
  {
    sub_7296C0(v6);
    v2 = sub_7401F0(a1);
    sub_729730(v6[0]);
    result = &qword_4F077B4;
    if ( qword_4F077B4 )
    {
      result = (__int64 *)sub_73C570(*(_QWORD *)(v4 + 120), 1, -1);
      *(_QWORD *)(v4 + 120) = result;
    }
  }
  *(_QWORD *)(v4 + 184) = v2;
  return result;
}
