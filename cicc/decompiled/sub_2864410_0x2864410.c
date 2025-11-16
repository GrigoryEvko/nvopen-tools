// Function: sub_2864410
// Address: 0x2864410
//
void __fastcall sub_2864410(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, int a5)
{
  __int64 v7; // rbx
  __int64 v8; // r9
  __int64 v9; // [rsp+0h] [rbp-40h]
  int v10; // [rsp+Ch] [rbp-34h]

  v9 = *(unsigned int *)(a4 + 48);
  if ( *(_DWORD *)(a4 + 48) )
  {
    v7 = 0;
    do
    {
      v8 = v7++;
      v10 = a5;
      sub_2863B50(a1, a2, a3, (__int64 *)a4, a5, v8, 0);
      a5 = v10;
    }
    while ( v9 != v7 );
  }
  if ( *(_QWORD *)(a4 + 32) == 1 )
    sub_2863B50(a1, a2, a3, (__int64 *)a4, a5, -1, 1);
}
