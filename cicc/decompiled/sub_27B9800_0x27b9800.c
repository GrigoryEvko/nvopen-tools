// Function: sub_27B9800
// Address: 0x27b9800
//
void __fastcall sub_27B9800(__int64 *a1, __int64 a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v4; // rax
  unsigned __int64 *v6; // rdx
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // [rsp-48h] [rbp-48h]
  __int64 *v15; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    v4 = 0;
    v6 = a3 - 3;
    v10 = *a1;
    if ( a3 )
      v4 = (__int64)v6;
    v14 = v4;
    if ( !(unsigned __int8)sub_B19DB0(v10, a2, v4) )
    {
      v11 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      {
        v12 = *(__int64 **)(a2 - 8);
        v15 = &v12[v11];
      }
      else
      {
        v15 = (__int64 *)a2;
        v12 = (__int64 *)(a2 - v11 * 8);
      }
      while ( v15 != v12 )
      {
        v13 = *v12;
        v12 += 4;
        sub_27B9800(a1, v13, a3, a4);
      }
      sub_B44550((_QWORD *)a2, *(_QWORD *)(v14 + 40), a3, a4);
    }
  }
}
