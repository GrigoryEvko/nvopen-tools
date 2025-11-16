// Function: sub_B97C00
// Address: 0xb97c00
//
void __fastcall sub_B97C00(__int64 a1, int a2, __int64 a3)
{
  char *v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rsi
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // rdx
  unsigned __int8 *v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // r14
  int v12; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = (char *)&v12;
  v13[0] = a3;
  v12 = a2;
  sub_B96E90((__int64)v13, a3, 1);
  v4 = *(unsigned int *)(a1 + 8);
  v5 = v4 + 1;
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v11 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 > (unsigned __int64)&v12 )
    {
      sub_B97B00((__int64 *)a1, v5);
      v4 = *(unsigned int *)(a1 + 8);
      v6 = *(_QWORD *)a1;
      v7 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      if ( (unsigned __int64)&v12 >= v11 + 16 * v4 )
      {
        sub_B97B00((__int64 *)a1, v5);
        v4 = *(unsigned int *)(a1 + 8);
        v6 = *(_QWORD *)a1;
      }
      else
      {
        sub_B97B00((__int64 *)a1, v5);
        v6 = *(_QWORD *)a1;
        v4 = *(unsigned int *)(a1 + 8);
        v3 = (char *)v13 + *(_QWORD *)a1 - v11;
      }
      v7 = *(_DWORD *)(a1 + 8);
    }
  }
  else
  {
    v6 = *(_QWORD *)a1;
    v7 = *(_DWORD *)(a1 + 8);
  }
  v8 = v6 + 16 * v4;
  if ( v8 )
  {
    *(_DWORD *)v8 = *(_DWORD *)v3;
    v9 = (unsigned __int8 *)*((_QWORD *)v3 + 1);
    *(_QWORD *)(v8 + 8) = v9;
    if ( v9 )
    {
      sub_B976B0((__int64)(v3 + 8), v9, v8 + 8);
      *((_QWORD *)v3 + 1) = 0;
    }
    v7 = *(_DWORD *)(a1 + 8);
  }
  v10 = v13[0];
  *(_DWORD *)(a1 + 8) = v7 + 1;
  if ( v10 )
    sub_B91220((__int64)v13, v10);
}
