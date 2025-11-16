// Function: sub_31EA6F0
// Address: 0x31ea6f0
//
void __fastcall sub_31EA6F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // r13
  _QWORD *v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // r13
  __int64 v14; // rsi
  void (__fastcall *v15)(__int64, __int64, _QWORD); // r14
  __int64 v16; // rax
  __int64 *v17; // [rsp+0h] [rbp-50h]
  __int64 v18; // [rsp+8h] [rbp-48h]
  _QWORD *v19; // [rsp+8h] [rbp-48h]
  _QWORD v20[8]; // [rsp+10h] [rbp-40h] BYREF

  v18 = *(_QWORD *)(a3 + 8);
  v6 = sub_AE5020(a2, v18);
  v7 = sub_9208B0(a2, v18);
  v20[1] = v8;
  v20[0] = ((1LL << v6) + ((unsigned __int64)(v7 + 7) >> 3) - 1) >> v6 << v6;
  if ( sub_CA1930(v20) )
  {
    sub_31E9900(a2, a3, a1, 0, 0, a4);
  }
  else if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 18LL) )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 536LL))(*(_QWORD *)(a1 + 224), 0, 1);
  }
  if ( a4 )
  {
    if ( *(_DWORD *)(a4 + 16) )
    {
      v9 = *(_QWORD **)(a4 + 8);
      v10 = 4LL * *(unsigned int *)(a4 + 24);
      v19 = &v9[v10];
      if ( v9 != &v9[v10] )
      {
        while ( 1 )
        {
          v11 = v9;
          if ( *v9 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v9 += 4;
          if ( v19 == v9 )
            return;
        }
        while ( v11 != v19 )
        {
          v12 = (__int64 *)v11[1];
          v17 = &v12[*((unsigned int *)v11 + 4)];
          while ( v17 != v12 )
          {
            v13 = *(_QWORD *)(a1 + 224);
            v14 = *v12++;
            v15 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v13 + 208LL);
            v16 = sub_31DB510(a1, v14);
            v15(v13, v16, 0);
          }
          v11 += 4;
          if ( v11 == v19 )
            break;
          while ( *v11 > 0xFFFFFFFFFFFFFFFDLL )
          {
            v11 += 4;
            if ( v19 == v11 )
              return;
          }
        }
      }
    }
  }
}
