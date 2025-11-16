// Function: sub_38EC400
// Address: 0x38ec400
//
__int64 __fastcall sub_38EC400(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r13
  unsigned int v5; // eax
  unsigned int v6; // r12d
  __int64 v7; // rdi
  int v8; // edx
  char v9; // al
  signed __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int *v13; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-40h] BYREF
  char v15; // [rsp+20h] [rbp-30h]
  char v16; // [rsp+21h] [rbp-2Fh]

  v2 = sub_3909290(*a1 + 144);
  v3 = *a1;
  v4 = v2;
  if ( !*(_BYTE *)(*a1 + 845) )
  {
    if ( (unsigned __int8)sub_38E36C0(v3) )
      return 1;
    v3 = *a1;
  }
  v14[0] = 0;
  LOBYTE(v5) = sub_38EB6A0(v3, (__int64 *)&v13, (__int64)v14);
  v6 = v5;
  if ( (_BYTE)v5 )
    return 1;
  v7 = *a1;
  v8 = *(_DWORD *)a1[1];
  if ( *v13 != 1 )
  {
    sub_38DDD30(*(_QWORD *)(v7 + 328), v13);
    return v6;
  }
  v9 = 8 * v8;
  v10 = *((_QWORD *)v13 + 2);
  if ( (unsigned int)(8 * v8) <= 0x3F && v10 > 0xFFFFFFFFFFFFFFFFLL >> (64 - v9) )
  {
    v11 = 1LL << (v9 - 1);
    if ( v10 < -v11 || v10 > v11 - 1 )
    {
      v16 = 1;
      v14[0] = "out of range literal value";
      v15 = 3;
      return (unsigned int)sub_3909790(v7, v4, v14, 0, 0);
    }
  }
  (*(void (__fastcall **)(_QWORD, signed __int64))(**(_QWORD **)(v7 + 328) + 424LL))(*(_QWORD *)(v7 + 328), v10);
  return v6;
}
