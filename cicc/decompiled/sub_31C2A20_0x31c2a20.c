// Function: sub_31C2A20
// Address: 0x31c2a20
//
__int64 __fastcall sub_31C2A20(__int64 a1, unsigned int a2, unsigned int a3, char a4)
{
  unsigned int v4; // r12d
  __int64 v5; // r13
  __int64 *v6; // r13
  unsigned int v7; // ebx
  unsigned int v8; // esi
  __int64 v10; // r14
  __int64 *v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned int v14; // esi
  __int64 v15; // [rsp+0h] [rbp-70h]
  unsigned int v17; // [rsp+10h] [rbp-60h]
  __int64 *v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+20h] [rbp-50h]
  unsigned int v22; // [rsp+2Ch] [rbp-44h]
  _QWORD v23[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = a2;
  v5 = *(_QWORD *)(a1 + 8);
  v19 = (__int64 *)(v5 + 8LL * *(unsigned int *)(a1 + 16));
  v6 = (__int64 *)(8LL * a2 + v5);
  v15 = 8LL * a2;
  if ( v6 != v19 )
  {
    v17 = 0;
    v7 = 0;
    v22 = 0;
    while ( *(_DWORD *)(a1 + 136) <= v4 || (*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL * (v4 >> 6)) & (1LL << v4)) == 0 )
    {
      v10 = *v6;
      v20 = sub_B43CA0(*(_QWORD *)(*v6 + 16)) + 312;
      if ( sub_318B630(v10) && (*(_DWORD *)(v10 + 8) != 37 || sub_318B6C0(v10)) )
      {
        if ( sub_318B670(v10) )
        {
          v10 = sub_318B680(v10);
        }
        else if ( *(_DWORD *)(v10 + 8) == 37 )
        {
          v10 = sub_318B6C0(v10);
        }
      }
      v11 = sub_318EB80(v10);
      v23[0] = sub_9208B0(v20, *v11);
      v23[1] = v12;
      v7 += sub_CA1930(v23);
      if ( v7 > a3 )
        break;
      ++v22;
      if ( v7 && a4 )
      {
        v13 = v17;
        if ( (v7 & (v7 - 1)) == 0 )
          v13 = v22;
        v17 = v13;
      }
      ++v6;
      ++v4;
      if ( v19 == v6 )
      {
        v14 = v17;
        if ( !a4 )
          v14 = v22;
        if ( v14 <= 1 )
          return 0;
        return *(_QWORD *)(a1 + 8) + v15;
      }
    }
    v8 = v17;
    if ( !a4 )
      v8 = v22;
    if ( v8 > 1 )
      return *(_QWORD *)(a1 + 8) + v15;
  }
  return 0;
}
