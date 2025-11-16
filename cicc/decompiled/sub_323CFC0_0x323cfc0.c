// Function: sub_323CFC0
// Address: 0x323cfc0
//
void __fastcall sub_323CFC0(__int64 a1)
{
  __int64 *v2; // r13
  __int64 v3; // r15
  __int64 v4; // rdi
  __int64 v5; // rbx
  void (__fastcall *v6)(__int64, _QWORD, _QWORD); // r15
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r8
  _QWORD *v13; // rbx
  _QWORD *v14; // r15
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned int v17; // eax
  _QWORD *v18; // rsi
  __int64 v19; // rax
  __int64 *v20; // [rsp+8h] [rbp-38h]

  if ( (unsigned __int16)sub_3220AA0(a1) > 4u )
  {
    v19 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
    sub_323CCF0(a1, *(_QWORD *)(v19 + 344));
  }
  else
  {
    v2 = *(__int64 **)(a1 + 1288);
    v3 = 4LL * *(unsigned int *)(a1 + 1296);
    v20 = &v2[v3];
    while ( v20 != v2 )
    {
      v4 = *(_QWORD *)(a1 + 8);
      v5 = *(_QWORD *)(v4 + 224);
      v6 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v5 + 176LL);
      v7 = sub_31DA6B0(v4);
      v6(v5, *(_QWORD *)(v7 + 272), 0);
      v8 = *(_QWORD *)(a1 + 8);
      if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(v8 + 200) + 544LL) - 42) > 1 )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v8 + 224) + 208LL))(
          *(_QWORD *)(v8 + 224),
          v2[1],
          0);
        v8 = *(_QWORD *)(a1 + 8);
      }
      v9 = v2[2];
      if ( (((__int64)v2 - *(_QWORD *)(a1 + 1288)) >> 5) + 1 == *(_DWORD *)(a1 + 1296) )
        v10 = *(unsigned int *)(a1 + 1440);
      else
        v10 = v2[6];
      v11 = v10 - v9;
      v12 = *(_QWORD *)(a1 + 1432) + 32 * v9;
      v13 = (_QWORD *)(v12 + 32 * v11);
      if ( v13 != (_QWORD *)v12 )
      {
        v14 = (_QWORD *)v12;
        do
        {
          sub_31DC9D0(v8, 3);
          v17 = sub_37291A0(a1 + 4840, *v14, 0, v15, v16);
          (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 424LL))(
            *(_QWORD *)(a1 + 8),
            v17,
            0,
            0);
          sub_31DCA50(*(_QWORD *)(a1 + 8));
          v18 = v14;
          v14 += 4;
          sub_3220AC0(a1, v18, *v2);
          v8 = *(_QWORD *)(a1 + 8);
        }
        while ( v13 != v14 );
      }
      v2 += 4;
      sub_31DC9D0(v8, 0);
    }
  }
}
