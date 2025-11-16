// Function: sub_24FCF50
// Address: 0x24fcf50
//
__int64 __fastcall sub_24FCF50(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  unsigned __int8 v7; // si
  __int64 v8; // r10
  __int64 v10; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F87C64 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_6;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F87C64);
  v7 = *(_BYTE *)(a1 + 169);
  v8 = *(_QWORD *)(v6 + 176);
  v10 = a1;
  v11[0] = a1;
  return sub_24FB8F0(
           a2,
           v7,
           v8,
           0,
           (__int64)sub_24FB3B0,
           (__int64)v11,
           (__int64 (__fastcall *)(__int64, unsigned __int8 *))sub_24FB410,
           (__int64)&v10);
}
