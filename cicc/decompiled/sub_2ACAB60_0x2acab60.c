// Function: sub_2ACAB60
// Address: 0x2acab60
//
__int64 __fastcall sub_2ACAB60(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rax
  __int64 result; // rax
  int v6; // ecx
  unsigned int v7; // esi
  int v8; // edx
  char v9; // dl
  __int64 v10; // rdx
  __int64 v11; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v12[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = (unsigned __int8)sub_2ABFC50(a1, (int *)a2, &v11) == 0;
  v4 = v11;
  if ( !v3 )
    return v11 + 8;
  v6 = *(_DWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v12[0] = v11;
  ++*(_QWORD *)a1;
  v8 = v6 + 1;
  if ( 4 * (v6 + 1) >= 3 * v7 )
  {
    v7 *= 2;
  }
  else if ( v7 - *(_DWORD *)(a1 + 20) - v8 > v7 >> 3 )
  {
    goto LABEL_5;
  }
  sub_2ACA8D0(a1, v7);
  sub_2ABFC50(a1, (int *)a2, v12);
  v8 = *(_DWORD *)(a1 + 16) + 1;
  v4 = v12[0];
LABEL_5:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *(_DWORD *)v4 != -1 || !*(_BYTE *)(v4 + 4) )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)v4 = *(_DWORD *)a2;
  v9 = *(_BYTE *)(a2 + 4);
  *(_QWORD *)(v4 + 8) = 0;
  *(_BYTE *)(v4 + 4) = v9;
  v10 = v4 + 40;
  result = v4 + 8;
  *(_QWORD *)(result + 8) = v10;
  *(_QWORD *)(result + 16) = 4;
  *(_DWORD *)(result + 24) = 0;
  *(_BYTE *)(result + 28) = 1;
  return result;
}
