// Function: sub_127F8B0
// Address: 0x127f8b0
//
__int64 __fastcall sub_127F8B0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // r12
  int v5; // ebx
  unsigned int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // [rsp+8h] [rbp-68h] BYREF
  const char *v16; // [rsp+10h] [rbp-60h] BYREF
  char v17; // [rsp+20h] [rbp-50h]
  char v18; // [rsp+21h] [rbp-4Fh]
  char v19[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v20; // [rsp+40h] [rbp-30h]

  v3 = sub_128F980();
  if ( !(unsigned __int8)sub_127B3A0(*a2) )
  {
    v5 = *(_DWORD *)(*(_QWORD *)v3 + 8LL) >> 8;
    if ( (unsigned int)sub_127B390() != v5 )
    {
      v18 = 1;
      v16 = "idxprom";
      v17 = 3;
      v6 = sub_127B390();
      v7 = sub_1644900(a1[5], v6);
      if ( v7 != *(_QWORD *)v3 )
      {
        if ( *(_BYTE *)(v3 + 16) > 0x10u )
        {
          v20 = 257;
          v8 = sub_15FE0A0(v3, v7, 0, v19, 0);
          v9 = a1[7];
          v3 = v8;
          if ( v9 )
          {
            v10 = (__int64 *)a1[8];
            sub_157E9D0(v9 + 40, v8);
            v11 = *(_QWORD *)(v3 + 24);
            v12 = *v10;
            *(_QWORD *)(v3 + 32) = v10;
            v12 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v3 + 24) = v12 | v11 & 7;
            *(_QWORD *)(v12 + 8) = v3 + 24;
            *v10 = *v10 & 7 | (v3 + 24);
          }
          sub_164B780(v3, &v16);
          v13 = a1[6];
          if ( v13 )
          {
            v15 = a1[6];
            sub_1623A60(&v15, v13, 2);
            if ( *(_QWORD *)(v3 + 48) )
              sub_161E7C0(v3 + 48);
            v14 = v15;
            *(_QWORD *)(v3 + 48) = v15;
            if ( v14 )
              sub_1623210(&v15, v14, v3 + 48);
          }
        }
        else
        {
          return sub_15A4750(v3, v7, 0);
        }
      }
    }
  }
  return v3;
}
