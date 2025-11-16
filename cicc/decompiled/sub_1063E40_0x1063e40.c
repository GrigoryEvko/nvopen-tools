// Function: sub_1063E40
// Address: 0x1063e40
//
char __fastcall sub_1063E40(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 v3; // r12
  int v4; // esi
  int v5; // eax
  __int64 v6; // rax
  char result; // al
  __int64 v8; // r14
  unsigned int v9; // ebx
  unsigned int i; // r12d
  __int64 *v11; // r13
  int v12; // eax
  bool v13; // al
  unsigned int v14; // r12d
  __int64 v15; // [rsp+8h] [rbp-68h]
  int v16; // [rsp+14h] [rbp-5Ch]
  __int64 v17; // [rsp+18h] [rbp-58h]
  __int64 *v18; // [rsp+20h] [rbp-50h]
  __int64 v19; // [rsp+28h] [rbp-48h] BYREF
  __int64 *v20; // [rsp+38h] [rbp-38h] BYREF

  v2 = *(_DWORD *)(a1 + 56);
  v19 = a2;
  if ( v2 )
  {
    v8 = *(_QWORD *)(a1 + 40);
    v9 = v2 - 1;
    v17 = sub_1061AC0();
    v15 = sub_1061AD0();
    v16 = 1;
    v18 = 0;
    for ( i = v9 & sub_1061E50(v19); ; i = v9 & v14 )
    {
      v11 = (__int64 *)(v8 + 8LL * i);
      result = sub_1061B40(v19, *v11);
      if ( result )
        break;
      if ( sub_1061B40(*v11, v17) )
      {
        v2 = *(_DWORD *)(a1 + 56);
        v3 = a1 + 32;
        if ( v18 )
          v11 = v18;
        v12 = *(_DWORD *)(a1 + 48);
        ++*(_QWORD *)(a1 + 32);
        v5 = v12 + 1;
        v20 = v11;
        if ( 4 * v5 >= 3 * v2 )
          goto LABEL_3;
        if ( v2 - (v5 + *(_DWORD *)(a1 + 52)) > v2 >> 3 )
          goto LABEL_5;
        v4 = v2;
        goto LABEL_4;
      }
      v13 = sub_1061B40(*v11, v15);
      if ( !v18 )
      {
        if ( !v13 )
          v11 = 0;
        v18 = v11;
      }
      v14 = v16 + i;
      ++v16;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 32);
    v3 = a1 + 32;
    v20 = 0;
LABEL_3:
    v4 = 2 * v2;
LABEL_4:
    sub_1063990(v3, v4);
    sub_1062220(v3, &v19, &v20);
    v5 = *(_DWORD *)(a1 + 48) + 1;
LABEL_5:
    *(_DWORD *)(a1 + 48) = v5;
    v6 = sub_1061AC0();
    if ( !sub_1061B40(*v20, v6) )
      --*(_DWORD *)(a1 + 52);
    result = (char)v20;
    *v20 = v19;
  }
  return result;
}
