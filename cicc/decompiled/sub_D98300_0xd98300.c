// Function: sub_D98300
// Address: 0xd98300
//
__int64 __fastcall sub_D98300(__int64 a1, __int64 a2)
{
  int v2; // r12d
  __int64 v3; // r14
  int v6; // edx
  unsigned int v7; // eax
  __int64 v8; // r12
  __int64 v9; // rsi
  int v11; // edi
  void *v12; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v13[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v14; // [rsp+18h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 152);
  if ( v2 )
  {
    v3 = *(_QWORD *)(a1 + 136);
    sub_D982A0(&v12, -4096, 0);
    v6 = v2 - 1;
    v7 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = v3 + 48LL * v7;
    v9 = *(_QWORD *)(v8 + 24);
    if ( v9 == a2 )
    {
LABEL_3:
      v12 = &unk_49DB368;
      if ( v14 && v14 != -8192 && v14 != -4096 )
        sub_BD60C0(v13);
      if ( v8 != *(_QWORD *)(a1 + 136) + 48LL * *(unsigned int *)(a1 + 152) )
        return *(_QWORD *)(v8 + 40);
    }
    else
    {
      v11 = 1;
      while ( v9 != v14 )
      {
        v7 = v6 & (v11 + v7);
        v8 = v3 + 48LL * v7;
        v9 = *(_QWORD *)(v8 + 24);
        if ( v9 == a2 )
          goto LABEL_3;
        ++v11;
      }
      if ( v9 && v9 != -4096 && v9 != -8192 )
      {
        v12 = &unk_49DB368;
        sub_BD60C0(v13);
      }
    }
  }
  return 0;
}
