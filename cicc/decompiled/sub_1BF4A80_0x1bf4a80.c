// Function: sub_1BF4A80
// Address: 0x1bf4a80
//
__int64 __fastcall sub_1BF4A80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v5; // rax
  bool v6; // zf
  unsigned int v7; // edx
  unsigned int v8; // ecx
  __int64 result; // rax
  __int64 v10; // [rsp+8h] [rbp-28h] BYREF
  __int64 v11[3]; // [rsp+18h] [rbp-18h] BYREF

  v10 = a3;
  v5 = sub_1BF18B0(a4);
  v6 = *(_QWORD *)(a1 + 8) == 0;
  v11[0] = (__int64)v5;
  if ( v6 )
  {
    result = 0;
  }
  else
  {
    if ( *(_DWORD *)(a4 + 40) == 1 || *(_DWORD *)(a4 + 8) > 1u )
    {
      v7 = *(_DWORD *)a1;
      v8 = dword_4FB99A0;
      result = 0;
      goto LABEL_5;
    }
    sub_1BF4360(*(__int64 **)(a1 + 16), v11, a1);
    result = 1;
  }
  v7 = *(_DWORD *)a1;
  v8 = dword_4FB99A0;
  if ( *(_DWORD *)a1 <= dword_5052308[0] || *(_DWORD *)(a4 + 40) == 1 || *(_DWORD *)(a4 + 8) > 1u )
  {
LABEL_5:
    if ( v7 <= v8 )
      return result;
  }
  sub_1BF46D0(*(__int64 **)(a1 + 16), v11, &v10);
  return 1;
}
