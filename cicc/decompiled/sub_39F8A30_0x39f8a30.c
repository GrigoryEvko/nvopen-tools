// Function: sub_39F8A30
// Address: 0x39f8a30
//
void __fastcall sub_39F8A30(__int64 a1, __int64 (__fastcall *a2)(__int64, _QWORD, _QWORD), __int64 a3, int a4, int a5)
{
  __int64 v5; // r9
  int v6; // ecx
  _QWORD *v9; // r14
  int v10; // eax
  int v11; // ebp
  _QWORD *v12; // rbx
  _QWORD *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // [rsp+8h] [rbp-40h]
  int v17; // [rsp+Ch] [rbp-3Ch]

  v5 = a4;
  v6 = 2 * a4 + 1;
  if ( v6 < a5 )
  {
    while ( 1 )
    {
      v11 = v6 + 1;
      v15 = 8LL * v6;
      v12 = (_QWORD *)(a3 + v15);
      if ( v6 + 1 < a5 )
      {
        v9 = (_QWORD *)(a3 + v15 + 8);
        v17 = v5;
        v16 = v6;
        v10 = a2(a1, *v12, *v9);
        v5 = v17;
        if ( v10 < 0 )
          v12 = v9;
        else
          v11 = v16;
      }
      else
      {
        v11 = v6;
      }
      v13 = (_QWORD *)(a3 + 8 * v5);
      if ( (int)a2(a1, *v13, *v12) >= 0 )
        break;
      v14 = *v13;
      v6 = 2 * v11 + 1;
      *v13 = *v12;
      *v12 = v14;
      if ( a5 <= v6 )
        break;
      v5 = v11;
    }
  }
}
