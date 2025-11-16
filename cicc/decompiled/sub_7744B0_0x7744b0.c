// Function: sub_7744B0
// Address: 0x7744b0
//
__int64 __fastcall sub_7744B0(__int64 a1, __int64 a2, __int16 *a3, _DWORD *a4)
{
  int v5; // r13d
  __int64 v6; // rbx
  int v7; // r14d
  __int64 result; // rax
  unsigned int v10; // [rsp+18h] [rbp-48h] BYREF
  int v11; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v12; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v13[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_729E00(dword_4F07508[0], &v12, v13, &v10, &v11);
  sub_620DE0(a3, v10);
  v5 = sub_8D27E0(a2);
  v6 = 16LL * *(unsigned __int8 *)(a2 + 160);
  v7 = sub_621000(a3, 0, (__int16 *)((char *)&unk_4F066C0 + v6), v5);
  result = sub_621000(a3, 0, (__int16 *)(v6 + 82864032), v5);
  if ( v7 > 0 || (int)result < 0 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xCB6u, (FILE *)dword_4F07508, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
    }
    *a4 = 0;
    return (__int64)a4;
  }
  return result;
}
