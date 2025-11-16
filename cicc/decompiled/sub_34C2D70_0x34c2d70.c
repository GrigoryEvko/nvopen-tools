// Function: sub_34C2D70
// Address: 0x34c2d70
//
__int64 __fastcall sub_34C2D70(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v8; // [rsp+0h] [rbp-50h] BYREF
  __int64 v9; // [rsp+8h] [rbp-48h]
  __int64 v10; // [rsp+10h] [rbp-40h]
  int v11; // [rsp+18h] [rbp-38h]

  v2 = 0;
  v3 = a2 + 320;
  sub_2E7A760(a2, 0);
  sub_34BA1B0((__int64)&v8, a2);
  sub_C7D6A0(*(_QWORD *)(a1 + 80), 16LL * *(unsigned int *)(a1 + 96), 8);
  v4 = v9;
  ++*(_QWORD *)(a1 + 72);
  ++v8;
  *(_QWORD *)(a1 + 80) = v4;
  v9 = 0;
  *(_QWORD *)(a1 + 88) = v10;
  v10 = 0;
  *(_DWORD *)(a1 + 96) = v11;
  v11 = 0;
  sub_C7D6A0(0, 0, 8);
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 328) + 8LL);
  if ( a2 + 320 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v5;
        v5 = *(_QWORD *)(v5 + 8);
        v2 |= sub_34C1120(a1, v6);
        if ( !*(_DWORD *)(v6 + 72) && !*(_BYTE *)(v6 + 217) )
          break;
        if ( v5 == v3 )
          return v2;
      }
      v2 = 1;
      sub_34BEF40(a1, v6);
    }
    while ( v5 != v3 );
  }
  return v2;
}
