// Function: sub_1852BE0
// Address: 0x1852be0
//
void __fastcall sub_1852BE0(const void *a1, size_t a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // r15
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rax
  size_t v10; // r9
  void *v11; // r10
  int v12; // eax
  void *v13; // [rsp+10h] [rbp-50h]
  size_t v14; // [rsp+18h] [rbp-48h]
  unsigned __int64 v15[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = (_QWORD *)(a3 + 8);
  v5 = *(_QWORD **)(a3 + 24);
  if ( (_QWORD *)(a3 + 8) != v5 )
  {
    while ( 1 )
    {
      v8 = (__int64 *)v5[7];
      if ( (__int64 *)v5[8] == v8 )
        goto LABEL_4;
      v15[0] = v5[4];
      v9 = *v8;
      v10 = *(_QWORD *)(v9 + 32);
      v11 = *(void **)(v9 + 24);
      if ( a2 != v10 )
        goto LABEL_3;
      v14 = *(_QWORD *)(v9 + 32);
      if ( !a2 )
        goto LABEL_4;
      v13 = *(void **)(v9 + 24);
      v12 = memcmp(v11, a1, a2);
      v11 = v13;
      v10 = v14;
      if ( v12 )
      {
LABEL_3:
        v7 = sub_1852A30(a4, (unsigned __int8 *)v11, v10);
        sub_18517D0((_QWORD *)(*(_QWORD *)v7 + 8LL), v15, 1);
LABEL_4:
        v5 = (_QWORD *)sub_220EF30(v5);
        if ( v4 == v5 )
          return;
      }
      else
      {
        v5 = (_QWORD *)sub_220EF30(v5);
        if ( v4 == v5 )
          return;
      }
    }
  }
}
