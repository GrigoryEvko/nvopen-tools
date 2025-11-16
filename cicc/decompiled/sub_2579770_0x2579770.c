// Function: sub_2579770
// Address: 0x2579770
//
__int64 __fastcall sub_2579770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rax
  char v8; // dl
  int v10; // ecx
  unsigned int v11; // esi
  int v12; // edx
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_256E050(a2, (__int64 *)a3, &v13) )
  {
    v5 = *(_QWORD *)a2;
    v6 = *(_QWORD *)(a2 + 8) + 24LL * *(unsigned int *)(a2 + 24);
    v7 = v13;
    v8 = 0;
    goto LABEL_3;
  }
  v10 = *(_DWORD *)(a2 + 16);
  v11 = *(_DWORD *)(a2 + 24);
  v7 = v13;
  ++*(_QWORD *)a2;
  v12 = v10 + 1;
  v14[0] = v7;
  if ( 4 * (v10 + 1) >= 3 * v11 )
  {
    v11 *= 2;
  }
  else if ( v11 - *(_DWORD *)(a2 + 20) - v12 > v11 >> 3 )
  {
    goto LABEL_6;
  }
  sub_2579470(a2, v11);
  sub_256E050(a2, (__int64 *)a3, v14);
  v12 = *(_DWORD *)(a2 + 16) + 1;
  v7 = v14[0];
LABEL_6:
  *(_DWORD *)(a2 + 16) = v12;
  if ( *(_QWORD *)v7 != -4096 || *(_QWORD *)(v7 + 8) != -4096 || *(_BYTE *)(v7 + 16) != 0xFF )
    --*(_DWORD *)(a2 + 20);
  *(_QWORD *)v7 = *(_QWORD *)a3;
  *(_QWORD *)(v7 + 8) = *(_QWORD *)(a3 + 8);
  *(_BYTE *)(v7 + 16) = *(_BYTE *)(a3 + 16);
  v5 = *(_QWORD *)a2;
  v6 = *(_QWORD *)(a2 + 8) + 24LL * *(unsigned int *)(a2 + 24);
  v8 = 1;
LABEL_3:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v7;
  *(_QWORD *)(a1 + 8) = v5;
  *(_QWORD *)(a1 + 24) = v6;
  *(_BYTE *)(a1 + 32) = v8;
  return a1;
}
